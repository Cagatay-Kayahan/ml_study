from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
TEST_SIZE = 0.20

proje_klasoru = Path(__file__).resolve().parent
sonuclar_klasoru = proje_klasoru / "results"
sonuclar_klasoru.mkdir(exist_ok=True)


def modelleri_egit_ve_test_et(veri_seti_adi, X, y):
    X = pd.get_dummies(X, drop_first=True, dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logistic_model = LogisticRegression(max_iter=5000)
    decision_tree_model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    random_forest_model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
    )

    logistic_model.fit(X_train_scaled, y_train)
    decision_tree_model.fit(X_train, y_train)
    random_forest_model.fit(X_train, y_train)

    modeller = [
        (
            "Logistic Regression",
            logistic_model.predict(X_train_scaled),
            logistic_model.predict(X_test_scaled),
        ),
        (
            "Decision Tree",
            decision_tree_model.predict(X_train),
            decision_tree_model.predict(X_test),
        ),
        (
            "Random Forest",
            random_forest_model.predict(X_train),
            random_forest_model.predict(X_test),
        ),
    ]

    sonuclar = []

    print("\n" + "=" * 70)
    print(veri_seti_adi)
    print(f"Kayıt sayısı: {len(X)} | Özellik sayısı: {X.shape[1]}")
    print("=" * 70)

    for model_adi, train_tahmin, test_tahmin in modeller:
        matrix = confusion_matrix(y_test, test_tahmin)

        sonuc = {
            "Veri Seti": veri_seti_adi,
            "Model": model_adi,
            "Train Accuracy": accuracy_score(y_train, train_tahmin),
            "Test Accuracy": accuracy_score(y_test, test_tahmin),
            "Precision": precision_score(
                y_test,
                test_tahmin,
                zero_division=0,
            ),
            "Recall": recall_score(
                y_test,
                test_tahmin,
                zero_division=0,
            ),
            "F1-score": f1_score(
                y_test,
                test_tahmin,
                zero_division=0,
            ),
            "TN": int(matrix[0, 0]),
            "FP": int(matrix[0, 1]),
            "FN": int(matrix[1, 0]),
            "TP": int(matrix[1, 1]),
        }

        sonuclar.append(sonuc)

        print(f"\n{model_adi}")
        print(f"Train Accuracy: {sonuc['Train Accuracy']:.4f}")
        print(f"Test Accuracy:  {sonuc['Test Accuracy']:.4f}")
        print(f"Precision:      {sonuc['Precision']:.4f}")
        print(f"Recall:         {sonuc['Recall']:.4f}")
        print(f"F1-score:       {sonuc['F1-score']:.4f}")
        print("Confusion Matrix:")
        print(matrix)

    return sonuclar


tum_sonuclar = []


# 1. IBM HR Analytics
ibm_yolu = (
    proje_klasoru
    / "data"
    / "ibm_hr"
    / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
)
ibm_veri = pd.read_csv(ibm_yolu)

ibm_y = ibm_veri["Attrition"].map({"No": 0, "Yes": 1})
ibm_X = ibm_veri.drop(
    columns=[
        "Attrition",
        "EmployeeNumber",
        "EmployeeCount",
        "Over18",
        "StandardHours",
    ],
    errors="ignore",
)

tum_sonuclar.extend(
    modelleri_egit_ve_test_et("IBM HR Analytics", ibm_X, ibm_y)
)


# 2. Telco Customer Churn
telco_yolu = (
    proje_klasoru
    / "data"
    / "telco_churn"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)
telco_veri = pd.read_csv(telco_yolu)

telco_veri["TotalCharges"] = pd.to_numeric(
    telco_veri["TotalCharges"],
    errors="coerce",
)
telco_veri = telco_veri.dropna().copy()

telco_y = telco_veri["Churn"].map({"No": 0, "Yes": 1})
telco_X = telco_veri.drop(
    columns=["Churn", "customerID"],
    errors="ignore",
)

tum_sonuclar.extend(
    modelleri_egit_ve_test_et(
        "Telco Customer Churn",
        telco_X,
        telco_y,
    )
)


# 3. Mushroom Classification
mushroom_yolu = proje_klasoru / "data" / "mushroom" / "mushrooms.csv"
mushroom_veri = pd.read_csv(mushroom_yolu)

# edible = 0, poisonous = 1
mushroom_y = mushroom_veri["class"].map({"e": 0, "p": 1})
mushroom_X = mushroom_veri.drop(columns=["class"])

tum_sonuclar.extend(
    modelleri_egit_ve_test_et(
        "Mushroom Classification",
        mushroom_X,
        mushroom_y,
    )
)


# Sonuçları kaydet
sonuclar_df = pd.DataFrame(tum_sonuclar)
sonuclar_df.to_csv(
    sonuclar_klasoru / "model_sonuclari.csv",
    index=False,
    encoding="utf-8-sig",
)

with open(
    sonuclar_klasoru / "llm_sonuclari.txt",
    "w",
    encoding="utf-8",
) as dosya:
    for veri_seti_adi in sonuclar_df["Veri Seti"].unique():
        dosya.write("=" * 70 + "\n")
        dosya.write(veri_seti_adi + "\n")
        dosya.write("=" * 70 + "\n\n")

        satirlar = sonuclar_df[
            sonuclar_df["Veri Seti"] == veri_seti_adi
        ]

        for _, satir in satirlar.iterrows():
            dosya.write(f"Model: {satir['Model']}\n")
            dosya.write(
                f"Train Accuracy: {satir['Train Accuracy']:.4f}\n"
            )
            dosya.write(
                f"Test Accuracy: {satir['Test Accuracy']:.4f}\n"
            )
            dosya.write(f"Precision: {satir['Precision']:.4f}\n")
            dosya.write(f"Recall: {satir['Recall']:.4f}\n")
            dosya.write(f"F1-score: {satir['F1-score']:.4f}\n")
            dosya.write(
                "Confusion Matrix: "
                f"[[{satir['TN']}, {satir['FP']}], "
                f"[{satir['FN']}, {satir['TP']}]]\n\n"
            )

print("\n" + "=" * 70)
print("TÜM TESTLER TAMAMLANDI")
print("=" * 70)
print("Sonuçlar results klasörüne kaydedildi.")

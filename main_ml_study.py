from pathlib import Path
import shutil
import sys
import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 1. Veri Setini Yükleme

DATASET_SLUG = "pavansubhasht/ibm-hr-analytics-attrition-dataset"
CSV_FILE_NAME = "WA_Fn-UseC_-HR-Employee-Attrition.csv"
TARGET_COLUMN = "Attrition"

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
CSV_PATH = DATA_DIR / CSV_FILE_NAME


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_dataset():
    print_section("1. Veri Setini Yükleme")
    DATA_DIR.mkdir(exist_ok=True)

    # Önce data klasöründe CSV var mı diye bakıyoruz.
    if CSV_PATH.exists():
        print(f"CSV dosyası bulundu: {CSV_PATH}")
        return pd.read_csv(CSV_PATH)

    # CSV yoksa Kaggle'dan indirmeyi deniyoruz.
    print("CSV dosyası bulunamadı. kagglehub ile indirme deneniyor...")

    try:
        import kagglehub

        download_folder = Path(kagglehub.dataset_download(DATASET_SLUG))
        csv_files = list(download_folder.rglob("*.csv"))

        if not csv_files:
            raise FileNotFoundError("İndirilen klasörde CSV dosyası bulunamadı.")

        matching_files = [file for file in csv_files if file.name == CSV_FILE_NAME]
        selected_csv = matching_files[0] if matching_files else csv_files[0]

        shutil.copyfile(selected_csv, CSV_PATH)
        print("Dataset indirildi ve data klasörüne kopyalandı.")
        print(f"Kullanılan CSV: {CSV_PATH}")

        return pd.read_csv(CSV_PATH)

    except Exception as error:
        print("Otomatik Kaggle indirme çalışmadı.")
        print(f"Hata: {error}")
        print("Lütfen CSV dosyasını manuel olarak şu konuma koy:")
        print(CSV_PATH)
        sys.exit(1)


def encode_attrition(dataframe):
    # Target değerini modelin anlayacağı 0/1 şekline çeviriyoruz.
    if TARGET_COLUMN not in dataframe.columns:
        print(f"Hedef kolon bulunamadı: {TARGET_COLUMN}")
        sys.exit(1)

    target = dataframe[TARGET_COLUMN].map({"Yes": 1, "No": 0})

    if target.isnull().any():
        print("Attrition kolonunda sadece Yes ve No değerleri olmalı.")
        sys.exit(1)

    return target.astype(int)


def add_feature_engineering(dataframe):
    # IBM HR kolonlarından yeni ve anlamlı feature'lar üretiyoruz.
    dataframe = dataframe.copy()
    added_features = []

    if {"MonthlyIncome", "YearsAtCompany"}.issubset(dataframe.columns):
        dataframe["IncomePerYearAtCompany"] = (
            dataframe["MonthlyIncome"] / (dataframe["YearsAtCompany"] + 1)
        )
        added_features.append("IncomePerYearAtCompany")

    satisfaction_columns = [
        "EnvironmentSatisfaction",
        "JobSatisfaction",
        "RelationshipSatisfaction",
    ]

    if set(satisfaction_columns).issubset(dataframe.columns):
        dataframe["TotalSatisfaction"] = dataframe[satisfaction_columns].sum(axis=1)
        added_features.append("TotalSatisfaction")

    return dataframe, added_features


def prepare_features(dataframe, use_feature_engineering=False):
    # Bu fonksiyon feature ve target ayırma + preprocessing işini toplar.
    working_df = dataframe.copy()
    added_features = []

    if use_feature_engineering:
        working_df, added_features = add_feature_engineering(working_df)

    target = encode_attrition(working_df)
    features = working_df.drop(columns=[TARGET_COLUMN])

    # Bu kolonlar modele fayda sağlamayan ID veya sabit kolonlardır.
    columns_to_drop = [
        "EmployeeNumber",
        "EmployeeCount",
        "Over18",
        "StandardHours",
    ]
    dropped_columns = [column for column in columns_to_drop if column in features.columns]
    features = features.drop(columns=dropped_columns)

    # Kategorik kolonları one-hot encoding ile sayısal hale getiriyoruz.
    categorical_columns = features.select_dtypes(
        include=["object", "str", "category"]
    ).columns.tolist()

    encoded_features = pd.get_dummies(
        features,
        columns=categorical_columns,
        drop_first=True,
        dtype=int,
    )

    return encoded_features, target, categorical_columns, dropped_columns, added_features


def make_train_test_split(features, target):
    # Stratify ile target oranını train ve test içinde benzer tutuyoruz.
    return train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )


def scale_features(X_train, X_test):
    # Logistic Regression için ölçek farklarını azaltmak iyi olur.
    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    return X_train_scaled, X_test_scaled


def train_logistic_regression(features, target, penalty="l2", c_value=1.0, solver="lbfgs"):
    X_train, X_test, y_train, y_test = make_train_test_split(features, target)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    model = LogisticRegression(
        penalty=penalty,
        C=c_value,
        solver=solver,
        max_iter=3000,
        random_state=42,
    )

    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)

    return {
        "model": model,
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "predictions": predictions,
        "train_score": train_score,
        "test_score": test_score,
        "difference": train_score - test_score,
    }


def print_train_test_scores(train_score, test_score):
    difference = train_score - test_score

    print(f"Train Score: {train_score:.4f}")
    print(f"Test Score: {test_score:.4f}")
    print(f"Difference: {difference:.4f}")

    if difference > 0.10:
        print("Yorum: Train skoru test skorundan çok yüksek. Overfitting olabilir.")
    elif train_score < 0.70 and test_score < 0.70:
        print("Yorum: İki skor da düşük. Underfitting olabilir.")
    else:
        print("Yorum: Train ve test skorlarının arası çok açık değil.")


def main():
    warnings.simplefilter("ignore", ConvergenceWarning)
    warnings.filterwarnings("ignore", message=".*penalty.*deprecated.*", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*Inconsistent values.*", category=UserWarning)

    dataframe = load_dataset()

    if dataframe.empty:
        print("Dataset boş görünüyor. CSV dosyasını kontrol et.")
        sys.exit(1)

    # 2. Temel Veri Kontrolü
    print_section("2. Temel Veri Kontrolü")
    print("İlk 5 satır:")
    print(dataframe.head())

    print("\nSatır ve kolon sayısı:")
    print(dataframe.shape)

    print("\nKolonlar:")
    print(dataframe.columns.tolist())

    print("\nDataset bilgisi:")
    dataframe.info()

    print("\nEksik değer kontrolü:")
    print(dataframe.isnull().sum())

    # 3. Target Analizi
    print_section("3. Target Analizi")
    print("Attrition değer sayıları:")
    print(dataframe[TARGET_COLUMN].value_counts())

    print("\nAttrition yüzde dağılımı:")
    print((dataframe[TARGET_COLUMN].value_counts(normalize=True) * 100).round(2))
    print("Yorum: Yes sınıfı daha az olduğu için target dengeli değil.")

    # 4. Preprocessing
    print_section("4. Preprocessing")
    features, target, categorical_columns, dropped_columns, _ = prepare_features(dataframe)

    print("Attrition encode edildi: Yes -> 1, No -> 0")

    print("\nSilinen kolonlar:")
    print(dropped_columns if dropped_columns else "Silinecek kolon bulunmadı.")

    print("\nKategorik kolonlar:")
    print(categorical_columns if categorical_columns else "Kategorik kolon bulunmadı.")

    print("\nOne-hot encoding sonrası feature tablosu:")
    print(features.shape)

    # 5. Train-Test Split
    print_section("5. Train-Test Split")
    X_train, X_test, y_train, y_test = make_train_test_split(features, target)
    print("Train feature shape:", X_train.shape)
    print("Test feature shape:", X_test.shape)
    print("Train target shape:", y_train.shape)
    print("Test target shape:", y_test.shape)

    # 6. Logistic Regression Modeli
    print_section("6. Logistic Regression Modeli")
    baseline_result = train_logistic_regression(features, target)
    print_train_test_scores(
        baseline_result["train_score"],
        baseline_result["test_score"],
    )

    # 7. Classification Metrikleri
    print_section("7. Classification Metrikleri")
    y_test = baseline_result["y_test"]
    predictions = baseline_result["predictions"]

    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"F1-score: {f1_score(y_test, predictions, zero_division=0):.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    print(
        "Yorum: Accuracy tek başına yeterli değil. Attrition=Yes daha az olduğu için "
        "recall ve F1-score değerlerine de bakmalıyız."
    )

    # 8. Feature Engineering
    print_section("8. Feature Engineering")
    engineered_features, engineered_target, _, _, added_features = prepare_features(
        dataframe,
        use_feature_engineering=True,
    )

    print("Eklenen yeni feature'lar:")
    print(added_features if added_features else "Yeni feature oluşturulamadı.")

    engineered_result = train_logistic_regression(engineered_features, engineered_target)

    score_comparison = pd.DataFrame(
        [
            {
                "Model": "Baseline",
                "Train Score": baseline_result["train_score"],
                "Test Score": baseline_result["test_score"],
                "Difference": baseline_result["difference"],
            },
            {
                "Model": "Feature Engineering",
                "Train Score": engineered_result["train_score"],
                "Test Score": engineered_result["test_score"],
                "Difference": engineered_result["difference"],
            },
        ]
    )

    print(score_comparison.round(4).to_string(index=False))
    print("Yorum: Yeni feature test skorunu artırıyorsa faydalı olabilir.")

    print("\nSayısal kolonların Attrition ile korelasyonu:")
    analysis_df = dataframe.copy()
    analysis_df["AttritionEncoded"] = encode_attrition(analysis_df)

    correlations = (
        analysis_df.corr(numeric_only=True)["AttritionEncoded"]
        .drop(labels=["AttritionEncoded"], errors="ignore")
        .sort_values(ascending=False)
    )
    print(correlations)
    print("Yorum: Korelasyon fikir verir ama tek başına kesin neden göstermez.")

    important_categorical_columns = [
        "Department",
        "JobRole",
        "OverTime",
        "MaritalStatus",
    ]

    for column in important_categorical_columns:
        if column in analysis_df.columns:
            print(f"\n{column} kolonuna göre Attrition oranı:")
            attrition_rate = (
                analysis_df.groupby(column)["AttritionEncoded"]
                .mean()
                .sort_values(ascending=False)
                .round(4)
            )
            print(attrition_rate)

    print("Yorum: Bu oranlar kategoriler arasındaki farkları görmek için kullanılır.")

    # 9. Coefficient Analizi
    print_section("9. Coefficient Analizi")
    coefficient_table = pd.DataFrame(
        {
            "Feature": baseline_result["X_train"].columns,
            "Coefficient": baseline_result["model"].coef_[0],
        }
    )

    coefficient_table["Absolute Coefficient"] = coefficient_table["Coefficient"].abs()
    coefficient_table = coefficient_table.sort_values(
        "Absolute Coefficient",
        ascending=False,
    )

    print("En etkili görünen ilk 15 feature:")
    print(coefficient_table.head(15).round(4).to_string(index=False))
    print("Yorum: Pozitif katsayı Attrition=1 ihtimalini artırır.")
    print("Yorum: Negatif katsayı Attrition=1 ihtimalini azaltır.")
    print("Yorum: Katsayılar kesin sebep-sonuç açıklaması değildir.")

    # 10. Regularization Denemeleri
    print_section("10. Regularization Denemeleri")
    c_values = [0.01, 0.1, 1, 10, 100]
    regularization_results = []

    X_train, X_test, y_train, y_test = make_train_test_split(features, target)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    experiments = [
        ("L1", "l1", "liblinear"),
        ("L2", "l2", "liblinear"),
    ]

    for label, penalty, solver in experiments:
        for c_value in c_values:
            try:
                model = LogisticRegression(
                    penalty=penalty,
                    C=c_value,
                    solver=solver,
                    max_iter=3000,
                    random_state=42,
                )
                model.fit(X_train_scaled, y_train)

                train_score = model.score(X_train_scaled, y_train)
                test_score = model.score(X_test_scaled, y_test)

                regularization_results.append(
                    {
                        "Penalty": label,
                        "C": c_value,
                        "Train Score": train_score,
                        "Test Score": test_score,
                        "Difference": train_score - test_score,
                    }
                )
            except Exception as error:
                print(f"{label}, C={c_value} atlandı. Hata: {error}")

    regularization_table = pd.DataFrame(regularization_results)
    print(regularization_table.round(4).to_string(index=False))

    print("Yorum: Küçük C daha güçlü regularization demektir.")
    print("Yorum: Büyük C daha zayıf regularization demektir.")
    print("Yorum: Train yüksek ama test düşükse overfitting olabilir.")


if __name__ == "__main__":
    main()

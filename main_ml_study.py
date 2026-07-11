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


# Bu fonksiyon sadece sonuçları ekrana yazdırır.
def sonucu_yazdir(model_adi, y_train, train_tahmin, y_test, test_tahmin):
    print(f"\n{model_adi}")
    print(f"Train Accuracy: {accuracy_score(y_train, train_tahmin):.4f}")
    print(f"Test Accuracy:  {accuracy_score(y_test, test_tahmin):.4f}")
    print(f"Precision:      {precision_score(y_test, test_tahmin, zero_division=0):.4f}")
    print(f"Recall:         {recall_score(y_test, test_tahmin, zero_division=0):.4f}")
    print(f"F1-score:       {f1_score(y_test, test_tahmin, zero_division=0):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, test_tahmin))


# ============================================================
# 1. IBM HR ANALYTICS
# ============================================================

print("\n" + "=" * 60)
print("IBM HR ANALYTICS")
print("=" * 60)

# Veri setini oku
ibm_veri = pd.read_csv(
    "data/ibm_hr/WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

# Hedef sütun: Attrition
ibm_y = ibm_veri["Attrition"].map({"No": 0, "Yes": 1})

# Tahminde kullanılacak sütunlar
ibm_X = ibm_veri.drop(
    columns=[
        "Attrition",
        "EmployeeNumber",
        "EmployeeCount",
        "Over18",
        "StandardHours",
    ]
)

# Metin sütunlarını sayıya çevir
ibm_X = pd.get_dummies(ibm_X, drop_first=True, dtype=int)

# Train ve test verilerini ayır
ibm_X_train, ibm_X_test, ibm_y_train, ibm_y_test = train_test_split(
    ibm_X,
    ibm_y,
    test_size=0.20,
    random_state=42,
    stratify=ibm_y,
)

# Logistic Regression için scaling
ibm_scaler = StandardScaler()
ibm_X_train_scaled = ibm_scaler.fit_transform(ibm_X_train)
ibm_X_test_scaled = ibm_scaler.transform(ibm_X_test)

# Modelleri oluştur
ibm_logistic = LogisticRegression(max_iter=5000)
ibm_tree = DecisionTreeClassifier(random_state=42)
ibm_forest = RandomForestClassifier(n_estimators=100, random_state=42)

# Modelleri eğit
ibm_logistic.fit(ibm_X_train_scaled, ibm_y_train)
ibm_tree.fit(ibm_X_train, ibm_y_train)
ibm_forest.fit(ibm_X_train, ibm_y_train)

# Tahminleri al
ibm_logistic_train_tahmin = ibm_logistic.predict(ibm_X_train_scaled)
ibm_logistic_test_tahmin = ibm_logistic.predict(ibm_X_test_scaled)

ibm_tree_train_tahmin = ibm_tree.predict(ibm_X_train)
ibm_tree_test_tahmin = ibm_tree.predict(ibm_X_test)

ibm_forest_train_tahmin = ibm_forest.predict(ibm_X_train)
ibm_forest_test_tahmin = ibm_forest.predict(ibm_X_test)

# Sonuçları göster
sonucu_yazdir(
    "Logistic Regression",
    ibm_y_train,
    ibm_logistic_train_tahmin,
    ibm_y_test,
    ibm_logistic_test_tahmin,
)

sonucu_yazdir(
    "Decision Tree",
    ibm_y_train,
    ibm_tree_train_tahmin,
    ibm_y_test,
    ibm_tree_test_tahmin,
)

sonucu_yazdir(
    "Random Forest",
    ibm_y_train,
    ibm_forest_train_tahmin,
    ibm_y_test,
    ibm_forest_test_tahmin,
)


# ============================================================
# 2. TELCO CUSTOMER CHURN
# ============================================================

print("\n" + "=" * 60)
print("TELCO CUSTOMER CHURN")
print("=" * 60)

# Veri setini oku
telco_veri = pd.read_csv(
    "data/telco_churn/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

# TotalCharges sütununu sayısal hale getir
telco_veri["TotalCharges"] = pd.to_numeric(
    telco_veri["TotalCharges"],
    errors="coerce",
)

# Dönüşmeyen boş satırları kaldır
telco_veri = telco_veri.dropna()

# Hedef sütun: Churn
telco_y = telco_veri["Churn"].map({"No": 0, "Yes": 1})

# Tahminde kullanılacak sütunlar
telco_X = telco_veri.drop(columns=["Churn", "customerID"])

# Metin sütunlarını sayıya çevir
telco_X = pd.get_dummies(telco_X, drop_first=True, dtype=int)

# Train ve test verilerini ayır
telco_X_train, telco_X_test, telco_y_train, telco_y_test = train_test_split(
    telco_X,
    telco_y,
    test_size=0.20,
    random_state=42,
    stratify=telco_y,
)

# Logistic Regression için scaling
telco_scaler = StandardScaler()
telco_X_train_scaled = telco_scaler.fit_transform(telco_X_train)
telco_X_test_scaled = telco_scaler.transform(telco_X_test)

# Modelleri oluştur
telco_logistic = LogisticRegression(max_iter=5000)
telco_tree = DecisionTreeClassifier(random_state=42)
telco_forest = RandomForestClassifier(n_estimators=100, random_state=42)

# Modelleri eğit
telco_logistic.fit(telco_X_train_scaled, telco_y_train)
telco_tree.fit(telco_X_train, telco_y_train)
telco_forest.fit(telco_X_train, telco_y_train)

# Tahminleri al
telco_logistic_train_tahmin = telco_logistic.predict(telco_X_train_scaled)
telco_logistic_test_tahmin = telco_logistic.predict(telco_X_test_scaled)

telco_tree_train_tahmin = telco_tree.predict(telco_X_train)
telco_tree_test_tahmin = telco_tree.predict(telco_X_test)

telco_forest_train_tahmin = telco_forest.predict(telco_X_train)
telco_forest_test_tahmin = telco_forest.predict(telco_X_test)

# Sonuçları göster
sonucu_yazdir(
    "Logistic Regression",
    telco_y_train,
    telco_logistic_train_tahmin,
    telco_y_test,
    telco_logistic_test_tahmin,
)

sonucu_yazdir(
    "Decision Tree",
    telco_y_train,
    telco_tree_train_tahmin,
    telco_y_test,
    telco_tree_test_tahmin,
)

sonucu_yazdir(
    "Random Forest",
    telco_y_train,
    telco_forest_train_tahmin,
    telco_y_test,
    telco_forest_test_tahmin,
)


# ============================================================
# 3. MUSHROOM CLASSIFICATION
# ============================================================

print("\n" + "=" * 60)
print("MUSHROOM CLASSIFICATION")
print("=" * 60)

# Veri setini oku
mushroom_veri = pd.read_csv("data/mushroom/mushrooms.csv")

# Hedef sütun: class
# e = edible = 0
# p = poisonous = 1
mushroom_y = mushroom_veri["class"].map({"e": 0, "p": 1})

# Tahminde kullanılacak sütunlar
mushroom_X = mushroom_veri.drop(columns=["class"])

# Metin sütunlarını sayıya çevir
mushroom_X = pd.get_dummies(mushroom_X, drop_first=True, dtype=int)

# Train ve test verilerini ayır
(
    mushroom_X_train,
    mushroom_X_test,
    mushroom_y_train,
    mushroom_y_test,
) = train_test_split(
    mushroom_X,
    mushroom_y,
    test_size=0.20,
    random_state=42,
    stratify=mushroom_y,
)

# Logistic Regression için scaling
mushroom_scaler = StandardScaler()
mushroom_X_train_scaled = mushroom_scaler.fit_transform(mushroom_X_train)
mushroom_X_test_scaled = mushroom_scaler.transform(mushroom_X_test)

# Modelleri oluştur
mushroom_logistic = LogisticRegression(max_iter=5000)
mushroom_tree = DecisionTreeClassifier(random_state=42)
mushroom_forest = RandomForestClassifier(n_estimators=100, random_state=42)

# Modelleri eğit
mushroom_logistic.fit(mushroom_X_train_scaled, mushroom_y_train)
mushroom_tree.fit(mushroom_X_train, mushroom_y_train)
mushroom_forest.fit(mushroom_X_train, mushroom_y_train)

# Tahminleri al
mushroom_logistic_train_tahmin = mushroom_logistic.predict(
    mushroom_X_train_scaled
)
mushroom_logistic_test_tahmin = mushroom_logistic.predict(
    mushroom_X_test_scaled
)

mushroom_tree_train_tahmin = mushroom_tree.predict(mushroom_X_train)
mushroom_tree_test_tahmin = mushroom_tree.predict(mushroom_X_test)

mushroom_forest_train_tahmin = mushroom_forest.predict(mushroom_X_train)
mushroom_forest_test_tahmin = mushroom_forest.predict(mushroom_X_test)

# Sonuçları göster
sonucu_yazdir(
    "Logistic Regression",
    mushroom_y_train,
    mushroom_logistic_train_tahmin,
    mushroom_y_test,
    mushroom_logistic_test_tahmin,
)

sonucu_yazdir(
    "Decision Tree",
    mushroom_y_train,
    mushroom_tree_train_tahmin,
    mushroom_y_test,
    mushroom_tree_test_tahmin,
)

sonucu_yazdir(
    "Random Forest",
    mushroom_y_train,
    mushroom_forest_train_tahmin,
    mushroom_y_test,
    mushroom_forest_test_tahmin,
)

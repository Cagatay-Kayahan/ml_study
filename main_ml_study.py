from pathlib import Path
import shutil
import sys
import warnings

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# Kütüphaneler ve sabit değerler
# ============================================================

DATASET_SLUG = "pavansubhasht/ibm-hr-analytics-attrition-dataset"
CSV_FILE_NAME = "WA_Fn-UseC_-HR-Employee-Attrition.csv"
TARGET_COLUMN = "Attrition"
RANDOM_STATE = 42

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
CSV_PATH = DATA_DIR / CSV_FILE_NAME


def print_section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ============================================================
# Dataset yükleme
# ============================================================


def load_dataset():
    DATA_DIR.mkdir(exist_ok=True)

    # Önce local CSV varsa onu kullanıyoruz.
    if CSV_PATH.exists():
        print(f"CSV dosyası bulundu: {CSV_PATH}")
        return pd.read_csv(CSV_PATH)

    # CSV yoksa kagglehub ile indirmeyi deniyoruz.
    print("CSV dosyası bulunamadı. kagglehub ile indirme deneniyor...")

    try:
        import kagglehub

        downloaded_folder = Path(kagglehub.dataset_download(DATASET_SLUG))
        csv_files = list(downloaded_folder.rglob("*.csv"))

        if not csv_files:
            raise FileNotFoundError("İndirilen klasörde CSV dosyası bulunamadı.")

        matching_files = [file for file in csv_files if file.name == CSV_FILE_NAME]
        selected_csv = matching_files[0] if matching_files else csv_files[0]

        shutil.copyfile(selected_csv, CSV_PATH)
        print("Dataset indirildi ve data klasörüne kopyalandı.")
        print(f"Kullanılan CSV: {CSV_PATH}")
        return pd.read_csv(CSV_PATH)

    except Exception as error:
        print("Dataset yüklenemedi.")
        print(f"Hata: {error}")
        print("Lütfen Kaggle'dan CSV dosyasını indirip şu konuma koy:")
        print(CSV_PATH)
        sys.exit(1)


# ============================================================
# İlk veri kontrolü
# ============================================================


def check_dataset(dataframe):
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

    print("\nAttrition değer sayıları:")
    print(dataframe[TARGET_COLUMN].value_counts())

    print("\nAttrition yüzde dağılımı:")
    print((dataframe[TARGET_COLUMN].value_counts(normalize=True) * 100).round(2))
    print("Yorum: Attrition=Yes sınıfı daha az olduğu için accuracy tek başına yeterli değildir.")


# ============================================================
# Preprocessing
# ============================================================


def split_features_and_target(dataframe):
    # Target kolonunu 0/1 değerine çeviriyoruz.
    if TARGET_COLUMN not in dataframe.columns:
        print(f"Hedef kolon bulunamadı: {TARGET_COLUMN}")
        sys.exit(1)

    target = dataframe[TARGET_COLUMN].map({"Yes": 1, "No": 0})

    if target.isnull().any():
        print("Attrition kolonunda sadece Yes ve No değerleri olmalı.")
        sys.exit(1)

    features = dataframe.drop(columns=[TARGET_COLUMN]).copy()

    # ID veya sabit değer gibi faydası düşük kolonları çıkarıyoruz.
    columns_to_drop = ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]
    existing_columns_to_drop = [column for column in columns_to_drop if column in features.columns]
    features = features.drop(columns=existing_columns_to_drop)

    print("Attrition encode edildi: Yes -> 1, No -> 0")
    print("Çıkarılan kolonlar:", existing_columns_to_drop if existing_columns_to_drop else "Yok")

    return features, target


def get_column_types(features):
    categorical_columns = features.select_dtypes(include=["object", "category", "str"]).columns.tolist()
    numeric_columns = features.select_dtypes(exclude=["object", "category", "str"]).columns.tolist()

    print("Sayısal kolon sayısı:", len(numeric_columns))
    print("Kategorik kolonlar:", categorical_columns)

    return numeric_columns, categorical_columns


def create_preprocessor(numeric_columns, categorical_columns):
    # ColumnTransformer, sayısal ve kategorik kolonlara farklı işlem uygular.
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_columns),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ]
    )


# ============================================================
# Model yardımcı fonksiyonları
# ============================================================


def create_model_pipeline(model, preprocessor):
    # Pipeline sayesinde preprocessing sadece train verisinden öğrenilir.
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate_model(model_name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)
    precision = precision_score(y_test, test_predictions, zero_division=0)
    recall = recall_score(y_test, test_predictions, zero_division=0)
    f1 = f1_score(y_test, test_predictions, zero_division=0)

    print_section(f"{model_name} Sonuçları")
    print(f"Train accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, test_predictions))

    print("\nClassification report:")
    print(classification_report(y_test, test_predictions, zero_division=0))

    print("Yorum: Attrition=Yes sınıfını yakalamak önemli olduğu için recall ve F1-score dikkatle incelenmeli.")

    return {
        "Model": model_name,
        "Train Accuracy": train_accuracy,
        "Test Accuracy": test_accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
    }


def run_cross_validation(models, features, target):
    print_section("Cross-validation Sonuçları")

    # StratifiedKFold, Yes/No oranını her fold içinde korumaya çalışır.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = []

    for model_name, model in models.items():
        scores = cross_val_score(model, features, target, cv=cv, scoring="accuracy")

        cv_results.append(
            {
                "Model": model_name,
                "CV Mean Accuracy": scores.mean(),
                "CV Std Accuracy": scores.std(),
            }
        )

    cv_results_df = pd.DataFrame(cv_results)
    print(cv_results_df.round(4).to_string(index=False))

    return cv_results_df


def tune_random_forest(random_forest_pipeline, X_train, y_train):
    print_section("Hyperparameter Tuning Sonuçları")

    # RandomizedSearchCV, tüm kombinasyonları değil seçilen rastgele denemeleri çalıştırır.
    parameter_grid = {
        "model__n_estimators": [100, 200, 300, 500],
        "model__max_depth": [None, 5, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", None],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=random_forest_pipeline,
        param_distributions=parameter_grid,
        n_iter=20,
        scoring="f1",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)

    print("En iyi parametreler:")
    print(search.best_params_)
    print(f"En iyi CV F1-score: {search.best_score_:.4f}")

    return search.best_estimator_, search.best_score_, search.best_params_


def print_final_comment(comparison_df, tuned_result):
    print_section("Sonuç Yorumu")
    best_default_model = comparison_df.sort_values("F1-score", ascending=False).iloc[0]

    print("Varsayılan modeller içinde F1-score'a göre en iyi model:")
    print(f"{best_default_model['Model']} - F1-score: {best_default_model['F1-score']:.4f}")

    print("\nTuned Random Forest test sonucu:")
    print(f"Test accuracy: {tuned_result['Test Accuracy']:.4f}")
    print(f"Recall: {tuned_result['Recall']:.4f}")
    print(f"F1-score: {tuned_result['F1-score']:.4f}")

    print(
        "\nGenel yorum: Bu problem binary classification problemidir. "
        "Attrition=Yes sınıfı az olduğu için sadece accuracy'ye bakmak yanıltıcı olabilir. "
        "Recall ve F1-score ayrılan çalışanları yakalamayı daha iyi gösterir."
    )


def main():
    warnings.simplefilter("ignore", ConvergenceWarning)

    print_section("Dataset Yükleme")
    dataframe = load_dataset()

    if dataframe.empty:
        print("Dataset boş görünüyor. CSV dosyasını kontrol et.")
        sys.exit(1)

    print_section("İlk Veri Kontrolü")
    check_dataset(dataframe)

    print_section("Preprocessing")
    features, target = split_features_and_target(dataframe)
    numeric_columns, categorical_columns = get_column_types(features)
    preprocessor = create_preprocessor(numeric_columns, categorical_columns)

    print_section("Train/Test Split")
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train dağılımı:")
    print(y_train.value_counts(normalize=True).round(4))
    print("y_test dağılımı:")
    print(y_test.value_counts(normalize=True).round(4))

    logistic_regression = create_model_pipeline(
        LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
        preprocessor,
    )
    decision_tree = create_model_pipeline(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        preprocessor,
    )
    random_forest = create_model_pipeline(
        RandomForestClassifier(random_state=RANDOM_STATE),
        preprocessor,
    )

    models = {
        "Logistic Regression": logistic_regression,
        "Decision Tree": decision_tree,
        "Random Forest": random_forest,
    }

    model_results = []
    for model_name, model in models.items():
        result = evaluate_model(model_name, model, X_train, X_test, y_train, y_test)
        model_results.append(result)

    print_section("Model Karşılaştırma")
    comparison_df = pd.DataFrame(model_results)
    print(comparison_df.round(4).to_string(index=False))

    run_cross_validation(models, features, target)

    tuned_random_forest, _, _ = tune_random_forest(random_forest, X_train, y_train)

    tuned_result = evaluate_model(
        "Tuned Random Forest",
        tuned_random_forest,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    print_section("Default Random Forest vs Tuned Random Forest")
    default_rf_result = comparison_df[comparison_df["Model"] == "Random Forest"].iloc[0].to_dict()
    rf_comparison = pd.DataFrame([default_rf_result, tuned_result])
    print(rf_comparison.round(4).to_string(index=False))

    print_final_comment(comparison_df, tuned_result)


if __name__ == "__main__":
    main()

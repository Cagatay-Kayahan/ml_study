# ML Cumulative Study

Bu proje, IBM HR Employee Attrition veri seti üzerinde çalışanların şirketten ayrılıp ayrılmayacağını tahmin etmek için hazırlanmış bir makine öğrenmesi çalışmasıdır.

## Proje Amacı

Amaç, `Attrition` kolonunu target olarak kullanarak binary classification modeli kurmaktır.

`Attrition` değeri kod içinde şu şekilde ele alınır:

```text
Yes -> 1
No -> 0
```

## Dataset

Kullanılan dataset:

**IBM HR Analytics Employee Attrition & Performance**

Kaggle slug:

```text
pavansubhasht/ibm-hr-analytics-attrition-dataset
```

Kod önce CSV dosyasını şu konumda arar:

```text
data/WA_Fn-UseC_-HR-Employee-Attrition.csv
```

Dosya yoksa `kagglehub` ile indirmeyi dener. `data/` klasörü `.gitignore` içinde olabilir; veri dosyasının GitHub'a eklenmesi zorunlu değildir.

## Kullanılan Modeller

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

## Kullanılan Metrikler

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Classification report
- Cross-validation mean accuracy
- Cross-validation std accuracy

Bu proje binary classification olduğu için accuracy, precision, recall, F1-score ve confusion matrix kullanılmıştır. MAE, MSE ve R² regression problemleri için daha uygundur.

## Hyperparameter Tuning

Random Forest modeli için `RandomizedSearchCV` kullanılmıştır.

Aranan bazı parametreler:

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`

Tuning sırasında scoring olarak `f1` kullanılır. Çünkü `Attrition=Yes` sınıfını yakalamak önemlidir ve accuracy tek başına yeterli olmayabilir.

## Dosya Yapısı

```text
ml_cumulative_study/
├── data/
├── main_ml_study.py
├── README.md
├── requirements.txt
└── .gitignore
```

Ana çalışma dosyası:

```text
main_ml_study.py
```

## Kurulum

Proje klasöründe gerekli paketleri yüklemek için:

```bash
pip install -r requirements.txt
```

## Çalıştırma

Proje klasöründe:

```bash
python main_ml_study.py
```

Terminal çıktısı bölüm bölüm ilerler:

- Dataset yükleme
- İlk veri kontrolü
- Preprocessing
- Train/test split
- Logistic Regression sonuçları
- Decision Tree sonuçları
- Random Forest sonuçları
- Model karşılaştırma
- Cross-validation sonuçları
- Hyperparameter tuning sonuçları
- Sonuç yorumu

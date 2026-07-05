# ML Cumulative Study

Bu proje, IBM HR Analytics Employee Attrition veri seti üzerinde temel makine öğrenmesi adımlarını tek bir çalışma dosyasında toplamak için hazırlanmıştır.

Ana çalışma dosyası:

```bash
main_ml_study.py
```

Projedeki amaç; veri okuma, veri kontrolü, preprocessing, model eğitimi, model değerlendirme, feature engineering, coefficient analizi ve regularization gibi adımları tek bir bütün akış içinde incelemektir.

## Dataset

Bu projede kullanılan veri seti:

**IBM HR Analytics Employee Attrition & Performance**

Kaggle dataset slug:

```text
pavansubhasht/ibm-hr-analytics-attrition-dataset
```

Hedef kolon:

```text
Attrition
```

Kod içinde `Attrition` kolonu şu şekilde sayısal hale getirilir:

```text
Yes -> 1
No -> 0
```

Bu sayede çalışan ayrılma durumu binary classification problemi olarak ele alınır.

## Proje Yapısı

```text
ml_cumulative_study/
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── main_ml_study.py
├── README.md
└── requirements.txt
```

## Kurulum

Gerekli paketleri yüklemek için proje kök klasöründe şu komut çalıştırılır:

```bash
pip install -r requirements.txt
```

## Çalıştırma

Ana çalışma dosyasını çalıştırmak için proje kök klasöründe şu komut kullanılır:

```bash
python main_ml_study.py
```

## CSV Dosyası

Script önce CSV dosyasını şu konumda arar:

```text
data/WA_Fn-UseC_-HR-Employee-Attrition.csv
```

Dosya bu konumda bulunmazsa `kagglehub` ile veri setini Kaggle üzerinden indirmeyi dener. İndirme başarılı olursa CSV dosyası `data` klasörüne kopyalanır ve ana script bu dosyayı kullanır.

Otomatik indirme çalışmazsa dataset Kaggle üzerinden manuel indirilebilir. Bu durumda CSV dosyasının şu konuma yerleştirilmesi gerekir:

```text
data/WA_Fn-UseC_-HR-Employee-Attrition.csv
```

## main_ml_study.py İçeriği

`main_ml_study.py` dosyasında aşağıdaki konular tek bir akış içinde yer alır:

- Dataset yükleme
- Temel veri kontrolü
- Eksik değer kontrolü
- Target analizi
- Feature ve target ayırma
- Gereksiz veya sabit kolonları çıkarma
- Kategorik kolonları one-hot encoding ile dönüştürme
- Train-test split
- Logistic Regression baseline model
- Train ve test skoru karşılaştırması
- Overfitting / underfitting yorumu
- Accuracy, precision, recall, F1-score metrikleri
- Confusion matrix ve classification report
- Feature engineering örneği
- Feature değerlendirme
- Logistic Regression coefficient analizi
- L1 ve L2 regularization denemesi
- Farklı C değerleriyle model karşılaştırması

## Kullanılan Temel Paketler

```text
pandas
scikit-learn
kagglehub
```

## Genel Not

Bu proje tek bir ana Python dosyası üzerinden ilerleyen kümülatif bir makine öğrenmesi çalışmasıdır. Yeni konular eklendikçe ana çalışma akışı `main_ml_study.py` içinde genişletilebilir.

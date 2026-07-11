# Model Sonuçları Özeti

Tüm veri setlerinde aynı ayarlar kullanılmıştır:

- Train/test oranı: %80 / %20
- Random state: 42
- Modeller: Logistic Regression, Decision Tree, Random Forest

## IBM HR Analytics

| Model | Train Accuracy | Test Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9022 | 0.8605 | 0.6154 | 0.3404 | 0.4384 |
| Decision Tree | 1.0000 | 0.7653 | 0.3103 | 0.3830 | 0.3429 |
| Random Forest | 1.0000 | 0.8333 | 0.4167 | 0.1064 | 0.1695 |

İlk gözlem: Logistic Regression en yüksek F1-score değerine sahiptir. Decision Tree ve Random Forest modellerinde train ve test skorları arasındaki fark overfitting ihtimalini göstermektedir.

## Telco Customer Churn

| Model | Train Accuracy | Test Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8043 | 0.8038 | 0.6476 | 0.5749 | 0.6091 |
| Decision Tree | 0.9988 | 0.7186 | 0.4701 | 0.4626 | 0.4663 |
| Random Forest | 0.9988 | 0.7896 | 0.6258 | 0.5187 | 0.5673 |

İlk gözlem: Logistic Regression test performansı ve F1-score bakımından en dengeli modeldir. Decision Tree belirgin biçimde overfitting göstermektedir.

## Mushroom Classification

| Model | Train Accuracy | Test Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Decision Tree | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

İlk gözlem: Üç model de ayrılan test verisinde hatasız sonuç vermiştir. Bu sonuç, veri setindeki sınıfların kullanılan özelliklerle çok güçlü biçimde ayrılabildiğini düşündürür; tek bir train/test bölünmesinin ötesinde doğrulama yapılmadan genelleme konusunda kesin hüküm verilmemelidir.

# Farklı Domainlerde Binary Classification Çalışması

Bu projede üç farklı domain üzerindeki binary classification problemleri, aynı makine öğrenmesi modelleri kullanılarak incelenmiştir.

## Veri Setleri

| Veri Seti | Domain | Hedef |
|---|---|---|
| IBM HR Analytics | İnsan kaynakları | Çalışan ayrılır mı? |
| Telco Customer Churn | Telekomünikasyon | Müşteri aboneliği bırakır mı? |
| Mushroom Classification | Biyoloji | Mantar zehirli mi? |

## Kullanılan Modeller

- Logistic Regression
- Decision Tree
- Random Forest

## Kullanılan Metrikler

- Train Accuracy
- Test Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Proje Akışı

1. Veri setlerini okuma
2. Hedef değişkeni belirleme
3. Gereksiz sütunları çıkarma
4. Kategorik verileri sayısallaştırma
5. Train/test ayırma
6. Modelleri eğitme
7. Tahmin alma
8. Sonuçları karşılaştırma
9. Sonuçları CSV ve metin dosyasına kaydetme

## Kurulum

```bash
pip install -r requirements.txt
```

## Çalıştırma

```bash
python main_ml_study.py
```

Çalışma tamamlandığında `results/` klasöründe şu dosyalar oluşur:

- `model_sonuclari.csv`
- `llm_sonuclari.txt`

`llm_sonuclari.txt`, sonuçların Gemini ve Gemma tarafından aynı prompt ile yorumlanması için hazırlanmıştır.

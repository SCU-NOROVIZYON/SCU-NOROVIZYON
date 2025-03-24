# Teknofest 2021 İnme Verisi ile Transfer Öğrenme ve Topluluk Modeli
<br><br>
## Proje Açıklaması

Bu proje, Teknofest 2021 inme verisini kullanarak iki sınıflı (inme var/yok) bir veri seti oluşturmayı, farklı transfer öğrenme modelleri ile çalışmayı ve en iyi performans veren iki modeli kullanarak topluluk öğrenme modelini oluşturmaya odaklanmaktadır.
<br><br>

## Veri Seti

Kaynak: Teknofest 2021 inme verisi (https://acikveri.saglik.gov.tr/Home/DataSets?categoryId=10)

Sınıflar: İnme var / İnme yok

Fold Kullanımı: 3 farklı fold kullanılmış olup, en iyi performansı veren fold 52 seçilmiştir.

Veri Artırma (Augmentation) Yöntemleri kullanılmıştır.

Eğitim ve Test Ayrımı: Veriler %80 eğitim ve %20 test olarak ayrılmıştır.

Ekstra Test Seti: Kaggle'dan harici bir test seti (https://www.kaggle.com/datasets/felipekitamura/head-ct-hemorrhage) bulunarak model performansı bu veri seti ile de değerlendirilmiştir.
<br><br>
## Kullanılan Modeller ve Topluluk Öğrenme

6 farklı transfer öğrenme modeli özelleştirilerek eğitilmiştir.
Modellerin en iyi sonuç veren ağırlıkları kaydedilmiştir (model_weights klasörü).

Bu modeller arasından en iyi performans veren iki model seçilerek topluluk öğrenme modeli oluşturulmuştur.
<br><br>
## Kullanım Talimatları
### 1. Ortamı Hazırlama

Proje için gerekli kütüp haneleri requirements.txt dosyasından kurulabilir:
``pip install -r requirements.txt``
### 2. Model Eğitimi

6 transfer öğrenme modelini eğitmek için:

## Sonuçlar

En iyi bireysel model ve topluluk öğrenme modelinin karmaşıklık matrisleri (confusion matrix) oluşturulmuştur.

t-SNE grafikleri, fully connected katmanlardan sonra özniteliklerin nasıl ayrıştığını göstermek için eklenmiştir (result klasörü).

Kaggle'dan alınan harici bir veri seti ile de model test edilmiştir (sample klasörü).
<br><br>
## Katkıda Bulunma

Proje geliştirmelerine katkıda bulunmak için pull request gönderebilirsiniz. Ayrıca hatalar ve öneriler için issue oluşturabilirsiniz.

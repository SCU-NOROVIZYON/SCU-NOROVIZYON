# Teknofest 2021 İnme Verisi ile Transfer Öğrenme ve Topluluk Modeli
<br><br>
## Proje Açıklaması

Bu proje, Teknofest 2021 inme verisini kullanarak iki sınıflı (inme var/yok) bir veri seti oluşturmayı, farklı transfer öğrenme modelleri ile çalışmayı ve en iyi performans veren iki modeli kullanarak topluluk öğrenme modelini oluşturmaya odaklanmaktadır.
<br><br>

## Veri Seti

Kaynak: Teknofest 2021 inme verisi (https://acikveri.saglik.gov.tr/Home/DataSets?categoryId=10)

Sınıflar: İnme var / İnme yok

Fold Kullanımı:Çapraz validasyon ile 3 alt küme oluşturulmuş ve en iyi model ağırlıkları CV3 ile elde edilmiştir.CV3 ile topluluk öğrenmede (vgg16+mobilnetv3_large) %99.67 F1 skor bulunmuştur

Veri Artırma (Augmentation) Yöntemleri kullanılmıştır.

Eğitim ve Test Ayrımı: Veriler %80 eğitim ve %20 test olarak ayrılmıştır.

Ekstra Test Seti: Kaggle'dan harici bir test seti (https://www.kaggle.com/datasets/felipekitamura/head-ct-hemorrhage) bulunarak model performansı bu veri seti ile de değerlendirilmiştir.
<br><br>
## Kullanılan Modeller ve Topluluk Öğrenme

6 farklı transfer öğrenme modeli özelleştirilerek eğitilmiştir.
Modellerin en iyi sonuç veren ağırlıkları kaydedilmiştir.
<br><br>
📂 [model_weights/](model_weights/)

Bu modeller arasından en iyi performans veren iki model seçilerek topluluk öğrenme modeli oluşturulmuştur.
<br><br>
## Kullanım Talimatları
### 1. Ortamı Hazırlama

Proje için gerekli kütüp haneleri requirements.txt dosyasından kurulabilir:
```
pip install -r requirements.txt
```
### 2. Model Eğitimi
6 transfer öğrenme modelini eğitmek için aşağıdaki Python dosyalarından ilgili olanı çalıştırabilirsiniz:

```
python ResNet50_Train.py
python VGG16_Train.py
python DenseNet121_Train.py
python EfficientNetB3_Train.py
python MobileNetV3_Large_Train.py
python Resnet-Inceptionv2_Train.py
```
### 3. Topluluk Öğrenme Modelini Oluşturma

En iyi performans veren iki modeli kullanarak topluluk öğrenme modelini oluşturmak için:
```
python topluluk_ogrenme_vgg_mobilenet.py
```
### 4. Harici Veri Seti ile Test Etme

Kaggle'dan bulunan harici veri setiyle modeli test etmek için:
```
python external_test.py
```

### 5. Örnek Çalıştırma

Bir inme olan ve olmayan fotoğraf için tahmin almak amacıyla:
<br><br>
📂 [sample/](sample/) - Modelin nasıl çalıştığını gösteren örnekler 
```
python sample/sample.py
```

## Sonuçlar

En iyi bireysel model ve topluluk öğrenme modelinin karmaşıklık matrisleri (confusion matrix) oluşturulmuştur.

t-SNE grafikleri, fully connected katmanlardan sonra özniteliklerin nasıl ayrıştığını göstermek için eklenmiştir.
<br><br>

📂 [results/](results/)

Kaggle'dan alınan harici bir veri seti ile de model test edilmiştir (sample klasörü).
<br><br>
## Katkıda Bulunma

Proje geliştirmelerine katkıda bulunmak için pull request gönderebilirsiniz. Ayrıca hatalar ve öneriler için issue oluşturabilirsiniz.

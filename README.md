💎 Elmas Fiyat Tahmini: Çoklu Doğrusal Regresyon ve Flask Projesi
Bu proje, Makine Öğrenmesi (BLG-407) dersi kapsamında Çoklu Doğrusal Regresyon (Multiple Linear Regression) teknikleri kullanılarak geliştirilmiştir. Projenin temel amacı, elmasların fiziksel özelliklerine (karat, kesim, renk, berraklık vb.) dayanarak fiyat tahmini yapan bir yapay zeka modeli eğitmek ve bu modeli bir web arayüzü ile son kullanıcıya sunmaktır.

Not: Proje kapsamında Geriye Doğru Eleme (Backward Elimination) yönteminin başarısını simüle etmek amacıyla veri setine yapay (dummy) değişkenler eklenmiş ve istatistiksel analiz (P-Value) sonucunda bu değişkenler başarıyla elenmiştir.

📂 Proje İçeriği ve Dosya Yapısı
Proje3_Regresyon.ipynb: Veri analizi, veri ön işleme, model eğitimi, Backward Elimination adımları ve model değerlendirme metriklerinin bulunduğu Jupyter Notebook dosyası.

app.py: Eğitilen modeli yükleyen ve kullanıcıdan alınan verilerle tahmin yapan Flask tabanlı web sunucusu kodları.

templates/index.html: Kullanıcının elmas özelliklerini girebileceği ve tahmin sonucunu görebileceği web arayüzü tasarımı.

elmas_modeli.pkl: Python ile eğitilmiş ve kaydedilmiş makine öğrenmesi modeli.

🚀 Kurulum ve Çalıştırma
Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz.

1. Gerekli Kütüphanelerin Yüklenmesi
Aşağıdaki Python kütüphanelerinin yüklü olduğundan emin olun:

Bash

pip install pandas numpy scikit-learn matplotlib seaborn statsmodels flask joblib
2. Modeli Eğitme (Opsiyonel)
Eğer modeli sıfırdan eğitmek ve .pkl dosyasını yeniden oluşturmak isterseniz Proje3_Regresyon.ipynb dosyasını Jupyter Notebook veya Google Colab üzerinde çalıştırabilirsiniz.

3. Web Arayüzünü Başlatma
Terminal veya komut satırını açarak proje klasörüne gelin ve aşağıdaki komutu çalıştırın:

Bash

python app.py
Komutu çalıştırdıktan sonra tarayıcınızda http://127.0.0.1:5000/ adresine giderek uygulamayı kullanabilirsiniz.

📊 Veri Bilimi ve Modelleme Süreci
Bu projede Seaborn Diamonds veri seti kullanılmıştır. Süreç şu adımlardan oluşur:

1. Veri Ön İşleme (Data Preprocessing)
Kategorik Verilerin Dönüşümü: Modelin matematiksel işlem yapabilmesi için cut (kesim), color (renk) ve clarity (berraklık) gibi metinsel veriler One-Hot Encoding yöntemiyle 0 ve 1'lere dönüştürülmüştür.

Dummy Variable Tuzağı: Çoklu bağlantı (Multicollinearity) sorununu önlemek amacıyla drop_first=True parametresi kullanılarak her kategoriden bir sütun atılmıştır.

2. Backward Elimination (Geriye Doğru Eleme) Senaryosu
Dersin isterlerini karşılamak ve feature selection başarısını göstermek için veri setine kasıtlı olarak fiyatla ilişkisi olmayan rastgele sütunlar eklenmiştir:

Kuyumcu_Adi: Rastgele marka isimleri.

Sertifika_No: Rastgele üretilen sayılar.

OLS (Ordinary Least Squares) raporu incelendiğinde, bu sütunların P-değerlerinin (P-value) 0.05'ten büyük olduğu (istatistiksel olarak anlamsız oldukları) görülmüş ve modelden elenmiştir. Ayrıca, carat (ağırlık) ile çok yüksek korelasyona sahip olan x, y, z boyut bilgilerinden sadece x tutulmuş, diğerleri elenmiştir.

3. Model Başarısı
Model test veri seti üzerinde değerlendirilmiş ve aşağıdaki sonuçlar elde edilmiştir:

R² (Belirlilik Katsayısı): ~0.92 (Model veriyi %92 oranında açıklayabilmektedir.)

MAE (Ortalama Mutlak Hata): ~737 $

MSE (Ortalama Kare Hata): ~1.288.764

💻 Web Arayüzü (Flask)
Kullanıcı dostu bir arayüz ile modelin tahmin yeteneği sergilenmiştir. app.py içerisinde, formdan gelen veriler (Örn: "Ideal" kesim, "E" renk) arka planda modelin anlayacağı One-Hot vektör formatına manuel olarak çevrilir ve modelden tahmin istenir.

Örnek Kullanım:

Karat giriniz (Örn: 0.75)

Kesim, Renk ve Berraklık seçiniz.

Derinlik ve Tablo oranlarını giriniz.

"FİYATI HESAPLA" butonuna basarak tahmini dolar değerini görünüz.

Geliştirici: Özge Zara Özçelik Ders: BLG-407 Makine Öğrenmesi

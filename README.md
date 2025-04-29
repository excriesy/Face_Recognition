Face Recognition
Bu proje, Python ve OpenCV kullanarak yüz tanıma işlemlerini gerçekleştirmeyi amaçlamaktadır. Proje, bilinen yüzleri tanımlamak ve yeni yüzleri tanımak için temel bir yapı sunar.​

Özellikler
Yüz tanıma için yüz encodings'lerini kaydetme ve yükleme

Gerçek zamanlı yüz tanıma

Tanınan yüzleri çıktı klasörüne kaydetme​

Gereksinimler
Projenin çalışması için aşağıdaki Python kütüphanelerine ihtiyaç vardır:​

Python 3.x

OpenCV

face_recognition

NumPy​
GitHub
+7
GitHub
+7
GitHub
+7
GitHub
+10
GitHub
+10
GitHub
+10

Gerekli kütüphaneleri yüklemek için aşağıdaki komutu kullanabilirsiniz:​

bash
Kopyala
Düzenle
pip install -r requirements.txt
Kurulum
Bu depoyu klonlayın:​

bash
Kopyala
Düzenle
git clone https://github.com/excriesy/Face_Recognition.git
cd Face_Recognition
Gerekli kütüphaneleri yükleyin:​

bash
Kopyala
Düzenle
pip install -r requirements.txt
Kullanım
1. Yüz Encodings'lerini Kaydetme
save_encodings.py dosyasını çalıştırarak known_faces klasöründeki yüzlerin encodings'lerini oluşturabilirsiniz:​

bash
Kopyala
Düzenle
python save_encodings.py
Bu işlem, encodings.pkl dosyasını oluşturacaktır.​

2. Gerçek Zamanlı Yüz Tanıma
main.py dosyasını çalıştırarak gerçek zamanlı yüz tanıma işlemini başlatabilirsiniz:​
GitHub

bash
Kopyala
Düzenle
python main.py
Bu işlem sırasında, tanınan yüzler outputs klasörüne kaydedilecektir.​

Dosya Yapısı
main.py: Gerçek zamanlı yüz tanıma işlemini başlatan ana dosya.

save_encodings.py: known_faces klasöründeki yüzlerin encodings'lerini oluşturur.

utils.py: Yardımcı fonksiyonları içerir.

encodings.pkl: Kaydedilmiş yüz encodings'lerini içerir.

known_faces/: Tanınması istenen kişilerin yüz görüntülerini içerir.

outputs/: Tanınan yüzlerin kaydedildiği klasör.

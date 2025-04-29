# Face Recognition

Bu proje, Python ve OpenCV kullanarak yüz tanıma işlemlerini gerçekleştirmeyi amaçlamaktadır.  
Bilinen yüzleri tanımlamak ve yeni yüzleri tanımak için temel bir yapı sunar.

## Özellikler

- Yüz encodings'lerini kaydetme ve yükleme  
- Gerçek zamanlı yüz tanıma  
- Tanınan yüzleri çıktı klasörüne kaydetme

## Gereksinimler

Projenin çalışabilmesi için aşağıdaki Python kütüphanelerine ihtiyaç vardır:

- Python 3.x  
- OpenCV  
- face_recognition  
- NumPy  

Gerekli kütüphaneleri yüklemek için:

```bash
pip install -r requirements.txt
```

## Kurulum

1. Depoyu klonlayın:

```bash
git clone https://github.com/excriesy/Face_Recognition.git
cd Face_Recognition
```

2. Bağımlılıkları yükleyin:

```bash
pip install -r requirements.txt
```

## Kullanım

### 1. Yüz Encodings'lerini Kaydetme

`save_encodings.py` dosyası, `known_faces/` klasöründeki kişilerin yüzlerini kodlayarak `encodings.pkl` dosyasını oluşturur:

```bash
python save_encodings.py
```

### 2. Gerçek Zamanlı Yüz Tanıma

`main.py` dosyasını çalıştırarak webcam üzerinden gerçek zamanlı yüz tanıma işlemini başlatabilirsiniz:

```bash
python main.py
```

Tanınan yüzler `outputs/` klasörüne otomatik olarak kaydedilir.

## Klasör Yapısı

```bash
Face_Recognition/
│
├── known_faces/        # Tanınacak kişilerin görüntüleri
├── outputs/            # Tanınan yüzlerin kaydedildiği klasör
├── encodings.pkl       # Kaydedilmiş yüz vektörleri
├── save_encodings.py   # Encodings oluşturma scripti
├── main.py             # Ana yüz tanıma uygulaması
├── utils.py            # Yardımcı fonksiyonlar
└── requirements.txt    # Gerekli kütüphaneler
```

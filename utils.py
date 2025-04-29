import cv2
import face_recognition
import pickle
from tkinter import Tk, filedialog
import numpy as np
import os
import time
import locale

# Türkçe karakter desteği için locale ayarı
try:
    locale.setlocale(locale.LC_ALL, 'tr_TR.UTF-8')
except:
    try:
        locale.setlocale(locale.LC_ALL, 'Turkish_Turkey.1254')  # Windows için
    except:
        print("[!] Türkçe karakter desteği sağlanamadı")

def select_video_file():
    """Video dosyası seçmek için dosya açma diyaloğu gösterir"""
    root = Tk()
    root.attributes('-topmost', True)  # Pencereyi en üstte tut
    root.withdraw()  # Ana pencereyi gizle
    file_path = filedialog.askopenfilename(
        title="Video seç",
        filetypes=[("Video Dosyaları","*.mp4 *.avi *.mov *.mkv *.wmv")]
    )
    root.destroy()  # Tk penceresini tamamen kapat
    return file_path

def select_image_file():
    """Resim dosyası seçmek için dosya açma diyaloğu gösterir"""
    root = Tk()
    root.attributes('-topmost', True)  # Pencereyi en üstte tut
    root.withdraw()  # Ana pencereyi gizle
    file_path = filedialog.askopenfilename(
        title="Fotoğraf seç",
        filetypes=[("Görüntü Dosyaları","*.jpg *.jpeg *.png *.bmp")]
    )
    root.destroy()  # Tk penceresini tamamen kapat
    return file_path

def load_encodings(pkl_path="encodings.pkl"):
    """Kaydedilmiş yüz verilerini yükler"""
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        return data["encodings"], data["names"]
    except FileNotFoundError:
        raise FileNotFoundError(f"'{pkl_path}' dosyası bulunamadı. Önce 'save_encodings.py' çalıştırın.")
    except Exception as e:
        raise Exception(f"Encodings yüklenirken hata: {e}")

def recognize_frame(frame, known_encodings, known_names, detection_scale=0.25, model="hog"):
    """
    Bir çerçevedeki yüzleri tanır ve konum+isim döndürür
    Parametreler:
    - frame: İşlenecek görüntü
    - known_encodings: Bilinen yüz kodlamaları listesi
    - known_names: Bilinen yüz isimlerinin listesi
    - detection_scale: Performans için görüntüyü küçültme oranı (varsayılan: 0.25)
    - model: Yüz tespiti için model ('hog' veya 'cnn')
    Dönüş:
    [(top, right, bottom, left, name), ...]
    """
    # Boş veya None frame kontrolü
    if frame is None or frame.size == 0:
        return []
    
    # Performans izleme başlat
    start_time = time.time()
    
    try:
        # Yüz tespiti için görüntüyü küçült
        small_frame = cv2.resize(frame, (0, 0), fx=detection_scale, fy=detection_scale)
        
        # BGR'den RGB'ye dönüştür (face_recognition kütüphanesi RGB beklediği için)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Yüz konumlarını bul
        face_locations = face_recognition.face_locations(rgb_small_frame, model=model)
        
        # Konumları orijinal boyuta göre ölçeklendir
        scale_factor = 1.0 / detection_scale
        scaled_face_locations = [(int(top * scale_factor), 
                                 int(right * scale_factor), 
                                 int(bottom * scale_factor), 
                                 int(left * scale_factor)) 
                                 for top, right, bottom, left in face_locations]
        
        # Yüz yoksa erken dön
        if not face_locations:
            return []
        
        # Yüz kodlamalarını al
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        # Tanınan yüzleri tutacak liste
        face_names = []
        
        # Her yüz için tanıma işlemini yap
        for face_encoding in face_encodings:
            # Bilinen tüm yüzlerle karşılaştır
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
            name = "Tanımlanamayan"  # Varsayılan isim
            
            # Eğer eşleşme varsa
            if True in matches:
                matched_indices = [i for i, match in enumerate(matches) if match]
                
                # En iyi eşleşmeyi bul
                if matched_indices:
                    # Yüz mesafelerini hesapla (düşük değer daha iyi eşleşme demektir)
                    face_distances = face_recognition.face_distance(
                        [known_encodings[i] for i in matched_indices], 
                        face_encoding
                    )
                    
                    # En düşük mesafeli eşleşmeyi al
                    best_match_index = np.argmin(face_distances)
                    name = known_names[matched_indices[best_match_index]]
            
            face_names.append(name)
        
        # İsim ve konum bilgilerini birleştir
        result = [(loc[0], loc[1], loc[2], loc[3], name) 
                 for loc, name in zip(scaled_face_locations, face_names)]
        
        # İşlem süresini hesapla
        processing_time = time.time() - start_time
        
        return result
    
    except Exception as e:
        print(f"[!] Kare işlenirken hata: {e}")
        return []

def draw_labels(frame, predictions):
    """
    Görüntüye yüz etiketlerini çizer
    Parametreler:
    - frame: Üzerine çizim yapılacak görüntü
    - predictions: recognize_frame() fonksiyonundan dönen [(top, right, bottom, left, name), ...] listesi
    Dönüş:
    - Etiketler çizilmiş görüntü
    """
    # Kopyasını oluştur
    labeled_frame = frame.copy()
    
    # Her tahmin için
    for top, right, bottom, left, name in predictions:
        # Yüz dikdörtgenini çiz
        cv2.rectangle(labeled_frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # İsim etiketi için arkaplan
        cv2.rectangle(labeled_frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        
        # İsim yazısını ekle
        cv2.putText(labeled_frame, name, (left + 6, bottom - 6), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
    
    return labeled_frame

def select_directory():
    """Dosya dizini seçmek için diyalog gösterir"""
    root = Tk()
    root.attributes('-topmost', True)  # Pencereyi en üstte tut
    root.withdraw()  # Ana pencereyi gizle
    dir_path = filedialog.askdirectory(title="Klasör seçin")
    root.destroy()  # Tk penceresini tamamen kapat
    return dir_path

def extract_face_encodings(image_path, model="hog"):
    """
    Bir resimden yüz kodlamalarını çıkarır
    Parametreler:
    - image_path: Resim dosyasının yolu
    - model: Yüz tespiti için model ('hog' veya 'cnn')
    Dönüş:
    - encodings: Yüz kodlamaları listesi
    - face_locations: Yüz konumları listesi
    """
    try:
        # Resmi oku
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"[!] '{image_path}' açılamadı")
            return [], []
        
        # BGR'den RGB'ye dönüştür
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Yüz tespiti yap
        face_locations = face_recognition.face_locations(rgb_image, model=model)
        
        if not face_locations:
            print(f"[!] '{image_path}' içinde yüz bulunamadı")
            return [], []
        
        # Yüz kodlamalarını al
        encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        return encodings, face_locations
        
    except Exception as e:
        print(f"[!] '{image_path}' işlenirken hata: {e}")
        return [], []

def save_encodings(encodings, names, output_file="encodings.pkl"):
    """
    Yüz kodlamalarını ve ilişkili isimleri dosyaya kaydeder
    Parametreler:
    - encodings: Yüz kodlamaları listesi
    - names: İsimler listesi
    - output_file: Kaydedilecek dosya adı
    """
    data = {"encodings": encodings, "names": names}
    
    try:
        with open(output_file, "wb") as f:
            pickle.dump(data, f)
        print(f"[+] Kodlamalar '{output_file}' dosyasına kaydedildi")
        return True
    except Exception as e:
        print(f"[!] Kodlamalar kaydedilirken hata: {e}")
        return False

def process_directory_for_encodings(directory, person_name=None, model="hog"):
    """
    Bir dizindeki tüm resimleri işleyerek yüz kodlamalarını çıkarır
    Parametreler:
    - directory: İşlenecek resim dizini
    - person_name: Resim kişi ismi (None ise klasör adı kullanılır)
    - model: Yüz tespiti için model ('hog' veya 'cnn')
    Dönüş:
    - encodings: Yüz kodlamaları listesi
    - names: İsimler listesi (her kodlamaya karşılık gelen)
    """
    # Resim uzantıları
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # Eğer kişi adı belirtilmemişse, klasör adını kullan
    if person_name is None:
        person_name = os.path.basename(directory)
    
    encodings = []
    names = []
    
    # Dizindeki tüm dosyaları işle
    for root, _, files in os.walk(directory):
        for file in files:
            # Sadece resim dosyalarını işle
            if any(file.lower().endswith(ext) for ext in image_extensions):
                file_path = os.path.join(root, file)
                print(f"[+] İşleniyor: {file_path}")
                
                # Yüz kodlamalarını çıkar
                face_encodings, face_locations = extract_face_encodings(file_path, model=model)
                
                # Bulunan her yüz için
                if face_encodings:
                    print(f"    - {len(face_encodings)} yüz bulundu")
                    encodings.extend(face_encodings)
                    names.extend([person_name] * len(face_encodings))
    
    return encodings, names

def merge_encodings(existing_file="encodings.pkl", new_encodings=[], new_names=[]):
    """
    Mevcut kodlama dosyasıyla yeni kodlamaları birleştirir
    Parametreler:
    - existing_file: Mevcut kodlama dosyası
    - new_encodings: Yeni kodlamalar listesi
    - new_names: Yeni isimler listesi
    Dönüş:
    - encodings: Birleştirilmiş kodlamalar listesi
    - names: Birleştirilmiş isimler listesi
    """
    try:
        # Mevcut dosya varsa yükle
        if os.path.exists(existing_file):
            with open(existing_file, "rb") as f:
                data = pickle.load(f)
            existing_encodings = data["encodings"]
            existing_names = data["names"]
            
            # Birleştir
            all_encodings = existing_encodings + new_encodings
            all_names = existing_names + new_names
            
            return all_encodings, all_names
        else:
            # Mevcut dosya yoksa sadece yenileri döndür
            return new_encodings, new_names
    
    except Exception as e:
        print(f"[!] Kodlamalar birleştirilirken hata: {e}")
        return [], []
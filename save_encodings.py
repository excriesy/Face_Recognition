import os
import face_recognition
import pickle
import cv2

def load_known_faces(known_faces_dir):
    known_encodings = []
    known_names = []
    
    # Klasör yoksa oluştur
    if not os.path.exists(known_faces_dir):
        os.makedirs(known_faces_dir)
        print(f"[!] '{known_faces_dir}' klasörü oluşturuldu. Lütfen içine yüz resimleri ekleyin.")
        return known_encodings, known_names
        
    # Kişi klasörlerini tarama
    person_folders = [d for d in os.listdir(known_faces_dir) 
                    if os.path.isdir(os.path.join(known_faces_dir, d))]
    
    if not person_folders:
        print(f"[!] '{known_faces_dir}' klasöründe hiç kişi klasörü bulunamadı.")
        return known_encodings, known_names
    
    total_images = 0
    processed_images = 0
    
    for person_name in person_folders:
        person_folder = os.path.join(known_faces_dir, person_name)
        person_images = [f for f in os.listdir(person_folder) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        total_images += len(person_images)
        person_encodings = 0
        
        print(f"[+] '{person_name}' için {len(person_images)} resim işleniyor...")
        
        for filename in person_images:
            file_path = os.path.join(person_folder, filename)
            
            try:
                # Resmi yükle
                image = cv2.imread(file_path)
                if image is None:
                    print(f"[!] Resim okunamadı: {file_path}")
                    continue
                    
                # BGR'dan RGB'ye dönüştür (face_recognition RGB bekler)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Yüzleri bul
                face_locations = face_recognition.face_locations(rgb_image)
                
                if not face_locations:
                    print(f"[!] Yüz bulunamadı: {file_path}")
                    continue
                
                # Yüz kodlamalarını al
                encodings = face_recognition.face_encodings(rgb_image, face_locations)
                
                if len(encodings) > 0:
                    known_encodings.append(encodings[0])
                    known_names.append(person_name)
                    person_encodings += 1
                    processed_images += 1
                
            except Exception as e:
                print(f"[!] Hata: {file_path}, {e}")
        
        print(f"[+] '{person_name}' için {person_encodings} yüz kaydedildi.")
    
    print(f"\n[+] Toplam {total_images} resimden {processed_images} yüz kaydedildi.")
    return known_encodings, known_names

def main():
    known_faces_dir = 'known_faces'
    output_file = "encodings.pkl"
    
    print("\n[+] Yüzler yükleniyor...")
    known_encodings, known_names = load_known_faces(known_faces_dir)
    
    if len(known_names) == 0:
        print("\n[!] Hiç yüz kaydedilmedi. İşlem sonlandırılıyor.")
        return
        
    print(f"\n[+] {len(known_names)} yüz başarıyla yüklendi.")
    
    # Verileri kaydet
    data = {"encodings": known_encodings, "names": known_names}
    with open(output_file, "wb") as f:
        pickle.dump(data, f)
    print(f"[+] Encodings dosyaya kaydedildi: {output_file}")
    
    # Benzersiz kişi sayısını göster
    unique_names = set(known_names)
    for name in unique_names:
        count = known_names.count(name)
        print(f"    - {name}: {count} yüz")

if __name__ == "__main__":
    main()
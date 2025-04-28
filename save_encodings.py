import os
import face_recognition
import pickle

def load_known_faces(known_faces_dir):
    known_encodings = []
    known_names = []

    for person_name in os.listdir(known_faces_dir):
        person_folder = os.path.join(known_faces_dir, person_name)

        if not os.path.isdir(person_folder):
            continue

        for filename in os.listdir(person_folder):
            file_path = os.path.join(person_folder, filename)

            try:
                image = face_recognition.load_image_file(file_path)
                encodings = face_recognition.face_encodings(image)

                if len(encodings) > 0:
                    known_encodings.append(encodings[0])
                    known_names.append(person_name)
                else:
                    print(f"Uyarı: Yüz bulunamadı -> {file_path}")
            except Exception as e:
                print(f"Hata: {file_path}, {e}")

    return known_encodings, known_names

def main():
    known_faces_dir = 'known_faces'

    print("[+] Yüzler yükleniyor...")
    known_encodings, known_names = load_known_faces(known_faces_dir)
    print(f"[+] {len(known_names)} yüz yüklendi.")

    data = {"encodings": known_encodings, "names": known_names}
    with open("encodings.pkl", "wb") as f:
        pickle.dump(data, f)
    print("[+] Encodings dosyaya kaydedildi: encodings.pkl")

if __name__ == "__main__":
    main()

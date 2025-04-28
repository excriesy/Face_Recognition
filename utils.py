import cv2
import face_recognition
import pickle
from tkinter import Tk, filedialog

# Tk penceresini tek seferlik başlat
_root = Tk()
_root.withdraw()

def select_video_file():
    return filedialog.askopenfilename(
        title="Video seç", filetypes=[("Video Dosyaları","*.mp4 *.avi *.mov")]
    )

def select_image_file():
    return filedialog.askopenfilename(
        title="Fotoğraf seç", 
        filetypes=[("Görüntü Dosyaları","*.jpg *.jpeg *.png")]
    )

def load_encodings(pkl_path="encodings.pkl"):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data["encodings"], data["names"]

def recognize_frame(frame, known_encodings, known_names):
    # Yüz tespiti yap
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb)
    
    if not face_locations:
        print("[!] Yüz bulunamadı.")
        return []  # Eğer yüz bulunmazsa boş bir liste döndürüyoruz

    # Yüz tanımlamaları (encodings) al
    face_encodings = face_recognition.face_encodings(rgb, face_locations)
    
    # Eğer yüz encodings'i mevcutsa
    face_names = []
    for encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, encoding)
        name = "Tanımlanamayan"
        
        # En iyi eşleşmeyi bul
        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]
        
        face_names.append(name)
    
    return list(zip(face_locations, face_names))


def draw_labels(frame, predictions, color=(0,255,0)):
    """
    predictions: list of (top,right,bottom,left,name)
    """
    for top, right, bottom, left, name in predictions:
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom-20), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left+2, bottom-4),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0), 1)
    return frame

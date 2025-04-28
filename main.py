import os
import cv2
from utils import (
    select_video_file, select_image_file,
    load_encodings, recognize_frame, draw_labels
)

def process_video(known_encodings, known_names):
    video_path = select_video_file()
    if not video_path:
        print("Video seçilmedi.")
        return
    print(f"[+] Video yolu: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[!] Video açılamadı.")
        return
    out_dir = "outputs"; os.makedirs(out_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(os.path.join(out_dir, f"out_{os.path.basename(video_path)}"),
                          fourcc, fps, (w,h))
    print("[+] Video işleniyor...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Video bitti veya okuma hatası.")
            break
        preds = recognize_frame(frame, known_encodings, known_names)
        print(f"[+] Prediksiyonlar: {preds}")  # debug çıktısı
        frame = draw_labels(frame, preds)
        out.write(frame)
    cap.release(); out.release()
    print("[+] Video kaydedildi.")


def process_camera(known_encodings, known_names):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] Kamera açılamadı.")
        return
    
    print("[+] Kamera başladı. Çıkmak için 'q' tuşuna basın.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Kamera hatası: Görüntü alınamadı.")
            break
        preds = recognize_frame(frame, known_encodings, known_names)
        if preds:
            print(f"[+] Prediksiyonlar: {preds}")  # debug çıktısı
        frame = draw_labels(frame, preds)
        cv2.imshow("Kamera", frame)
        
        # 'q' tuşuna basınca çıkış
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()



def process_image(known_encodings, known_names):
    img_path = select_image_file()
    if not img_path:
        print("Fotoğraf seçilmedi.")
        return
    print(f"[+] Fotoğraf yolu: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print("[!] Fotoğraf açılamadı.")
        return
    preds = recognize_frame(img, known_encodings, known_names)
    print(f"[+] Prediksiyonlar: {preds}")  # debug çıktısı
    img = draw_labels(img, preds)
    cv2.imshow("Fotoğraf", img)
    cv2.waitKey(0); cv2.destroyAllWindows()


def main():
    print("[+] Encodingler yükleniyor...")
    known_encodings, known_names = load_encodings()
    print(f"[+] {len(known_names)} yüz yüklendi.")
    print("Yapılacak işlemi seç:\n1) Video  2) Kamera  3) Fotoğraf")
    choice = input("Seçim (1/2/3): ").strip()
    if choice=="1": process_video(known_encodings, known_names)
    elif choice=="2": process_camera(known_encodings, known_names)
    elif choice=="3": process_image(known_encodings, known_names)
    else: print("Geçersiz seçim.")

if __name__=="__main__":
    main()

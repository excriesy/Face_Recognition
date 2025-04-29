import os
import cv2
import time
import numpy as np
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
    
    # Video yakalama
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[!] Video açılamadı.")
        return
    
    # Çıktı dizini oluştur
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    
    # Video özellikleri
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Performans için eğer video çok büyükse, boyutu küçült
    max_width = 1024  # Daha büyük ekran boyutu
    if w > max_width:
        scale_factor = max_width / w
        display_w = max_width
        display_h = int(h * scale_factor)
    else:
        display_w, display_h = w, h
    
    # Video yazıcı
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(out_dir, f"out_{os.path.basename(video_path)}")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    
    print("[+] Video işleniyor...")
    print("[+] Kontroller:")
    print("    - Space: Oynat/Duraklat")
    print("    - Sol/Sağ Ok: 10 kare geri/ileri")
    print("    - Q: Çıkış")
    
    # Video işleme değişkenleri
    frame_count = 0
    preds = []
    
    # Gerçek zamanlı işleme için frame atlama sayısını hesapla (fps'e göre)
    # Daha gerçek zamanlı olması için fps'e göre ayarla
    if fps > 30:
        process_frame = 6  # Yüksek fps için daha fazla atlama
    elif fps > 20:
        process_frame = 4
    else:
        process_frame = 2  # Düşük fps videolar için daha az atlama
    
    paused = False
    last_processed_frame = None
    frame = None  # İlk kareyi tanımla
    
    # İlerleme gösterimi için değişkenler
    start_time = time.time()
    last_update = 0
    processing_time = 0
    
    # Video oynatma fps kontrolü
    frame_time = 1.0 / fps if fps > 0 else 0.033  # Videonun doğal hızını korumak için
    last_frame_time = time.time()
    
    # Pencere oluştur ve en üstte görünmesini sağla
    window_name = "Video İşleme"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_w, display_h + 100)  # Kontrol paneli için ekstra alan
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    
    # Buton görüntüleri
    control_panel_height = 100
    button_width = 50
    button_height = 50
    
    # İlk kareyi oku
    ret, frame = cap.read()
    if not ret:
        print("[!] Video açılamadı veya boş.")
        return
        
    while True:
        current_time = time.time()
        
        # Eğer duraklatılmadıysa ve framelerin kendi hızında gösterilmesi için gerekli süre geçtiyse
        should_process_new_frame = not paused and (current_time - last_frame_time >= frame_time)
        
        if should_process_new_frame:
            ret, frame = cap.read()
            if not ret:
                print("\n[!] Video sonu veya okuma hatası.")
                break
            
            frame_count += 1
            last_frame_time = current_time
            
            # Her N karede bir yüz tanıma yap (performans için)
            if frame_count % process_frame == 0:
                process_start = time.time()
                # Tanıma kalitesini biraz artıralım
                preds = recognize_frame(frame, known_encodings, known_names, detection_scale=0.3)
                processing_time = time.time() - process_start
                last_processed_frame = frame.copy()
            
            # Etiketleri çiz
            labeled_frame = draw_labels(frame, preds)
            
            # Kaydet
            out.write(labeled_frame)
        else:
            # Duraklatıldığında son kareyi göster
            if last_processed_frame is not None:
                labeled_frame = last_processed_frame
            elif frame is not None:
                labeled_frame = draw_labels(frame, preds)
            else:
                # Hiç frame yoksa uygun bir hata mesajı göster
                print("[!] Gösterilecek kare yok.")
                break
        
        # Display için boyut ayarla
        if w > max_width:
            display_frame = cv2.resize(labeled_frame, (display_w, display_h))
        else:
            display_frame = labeled_frame
        
        # Bilgi çubuğu ekle - daha düzenli ve görsel
        info_bar = create_info_bar(
            frame_count, total_frames, progress=(frame_count / total_frames * 100) if total_frames > 0 else 0, 
            processing_time=processing_time, paused=paused, elapsed=current_time-start_time,
            width=display_frame.shape[1], height=control_panel_height
        )
        
        # Görüntüye bilgi çubuğunu ekle
        final_display = cv2.vconcat([display_frame, info_bar])
        
        # İşleme durumunu göster ve tuş kontrollerini al
        cv2.imshow(window_name, final_display)
        key = cv2.waitKey(1) & 0xFF
        
        # Tuş kontrolleri
        if key == ord('q'):  # Çıkış
            break
        elif key == 32:  # Space - Oynat/Duraklat
            paused = not paused
            if paused:
                print("\n[+] Video duraklatıldı")
            else:
                print("\n[+] Video oynatılıyor")
                last_frame_time = time.time()  # Düzgün zamanlama için sıfırla
        elif key == 83 and paused:  # Sağ ok - 10 kare ileri
            for _ in range(10):
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
            if ret:
                process_start = time.time()
                preds = recognize_frame(frame, known_encodings, known_names, detection_scale=0.3)
                processing_time = time.time() - process_start
                last_processed_frame = draw_labels(frame, preds)
        elif key == 81 and paused:  # Sol ok - 10 kare geri
            current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_pos - 10))
            frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            ret, frame = cap.read()
            if ret:
                process_start = time.time()
                preds = recognize_frame(frame, known_encodings, known_names, detection_scale=0.3)
                processing_time = time.time() - process_start
                last_processed_frame = draw_labels(frame, preds)
        
        # Terminal için ilerleme gösterimi (her saniyede bir güncelle)
        if current_time - last_update >= 1 and not paused:
            fps_value = frame_count / (current_time - start_time) if current_time > start_time else 0
            remaining = (total_frames - frame_count) / fps_value if fps_value > 0 else 0
            print(f"\r[+] İşleniyor: %{(frame_count / total_frames * 100):.1f} | FPS: {fps_value:.1f} | Kalan: {remaining:.1f}s | {frame_count}/{total_frames}", end="")
            last_update = current_time
    
    # Tamamlandı
    print(f"\n[+] Video işleme tamamlandı. Süre: {time.time() - start_time:.2f} saniye")
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[+] Video kaydedildi: {out_path}")


def create_info_bar(frame_count, total_frames, progress, processing_time, paused, elapsed, width, height):
    """Geliştirilmiş bilgi çubuğu oluştur"""
    # Koyu gri arkaplan oluştur
    info_bar = np.zeros((height, width, 3), dtype=np.uint8)
    info_bar[:] = (50, 50, 50)  # Koyu gri arkaplan
    
    # İlerleme çubuğu - daha görsel
    progress_height = 10
    bar_y = 15
    bar_width = int(width * progress / 100)
    
    # İlerleme arkaplanı
    cv2.rectangle(info_bar, (10, bar_y), (width-10, bar_y+progress_height), (100, 100, 100), -1)
    
    # İlerleme dolgusu
    cv2.rectangle(info_bar, (10, bar_y), (10 + bar_width, bar_y+progress_height), (0, 200, 0), -1)
    
    # İlerleme çerçevesi
    cv2.rectangle(info_bar, (10, bar_y), (width-10, bar_y+progress_height), (200, 200, 200), 1)
    
    # Bilgi metinleri - daha düzenli
    fps_rate = frame_count / elapsed if elapsed > 0 else 0
    status = "DURAKLATILDI" if paused else "OYNATILIYOR"
    
    # Sol bilgi paneli
    frame_info = f"Kare: {frame_count}/{total_frames} (%{progress:.1f})"
    cv2.putText(info_bar, frame_info, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Sağ bilgi paneli
    perf_info = f"İşlem: {processing_time*1000:.1f}ms | FPS: {fps_rate:.1f}"
    time_info = f"Süre: {elapsed:.1f}s | Durum: {status}"
    
    perf_text_size = cv2.getTextSize(perf_info, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
    cv2.putText(info_bar, perf_info, (width - perf_text_size[0] - 15, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    time_text_size = cv2.getTextSize(time_info, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
    cv2.putText(info_bar, time_info, (width - time_text_size[0] - 15, 75), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Kontrol tuşları bilgisi - daha görsel
    control_info = "SPACE: Oynat/Duraklat | ←/→: 10 kare | Q: Çıkış"
    cv2.putText(info_bar, control_info, (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
    
    # Kontrol butonları (görsel)
    if paused:
        # Oynat butonu (yeşil üçgen)
        cv2.circle(info_bar, (width//2, height//2), 15, (0, 150, 0), -1)
        pts = np.array([[width//2-5, height//2-8], [width//2-5, height//2+8], [width//2+8, height//2]], np.int32)
        cv2.fillPoly(info_bar, [pts], (255, 255, 255))
    else:
        # Duraklat butonu (iki dikey çubuk)
        cv2.circle(info_bar, (width//2, height//2), 15, (0, 0, 150), -1)
        cv2.rectangle(info_bar, (width//2-6, height//2-7), (width//2-2, height//2+7), (255, 255, 255), -1)
        cv2.rectangle(info_bar, (width//2+2, height//2-7), (width//2+6, height//2+7), (255, 255, 255), -1)
    
    return info_bar


def process_camera(known_encodings, known_names):
    # Kamera yakalama
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] Kamera açılamadı.")
        return
    
    print("[+] Kamera başladı. Çıkmak için 'q' tuşuna basın.")
    
    # Kamera ayarlarını optimize et - daha yüksek çözünürlük
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Gerçek zamanlı işleme ayarları
    process_frame = 3  # Her 3 karede bir yüz tanıma (daha sık)
    frame_count = 0
    preds = []
    last_time = time.time()
    fps_values = []
    processing_time = 0
    
    # Performans izleme
    frame_times = []  # Son 30 kare süresini izle
    
    # Pencere oluştur ve en üstte görünmesini sağla
    window_name = "Kamera - Yüz Tanıma"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 820)  # Kontrol çubuğu için ekstra alan
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    
    while True:
        # Kare zamanını ölç
        frame_start = time.time()
        
        ret, frame = cap.read()
        if not ret:
            print("[!] Kamera hatası: Görüntü alınamadı.")
            break
        
        frame_count += 1
        current_time = time.time()
        frame_time = current_time - frame_start
        
        # Son 30 karenin süresini tut (FPS hesaplama için)
        frame_times.append(frame_time)
        if len(frame_times) > 30:
            frame_times.pop(0)
        
        # Her N karede bir tanıma işlemi yap
        if frame_count % process_frame == 0:
            process_start = time.time()
            # Daha yüksek scale değeri ile daha iyi tanıma
            preds = recognize_frame(frame, known_encodings, known_names, detection_scale=0.4)
            processing_time = time.time() - process_start
        
        # Etiketleri çiz
        labeled_frame = draw_labels(frame, preds)
        
        # Kamera bilgi panelini ekle
        avg_frame_time = sum(frame_times) / len(frame_times) if frame_times else 0.033
        camera_info_panel = create_camera_info_panel(
            labeled_frame.shape[1], 100, preds, 
            fps=1.0/avg_frame_time if avg_frame_time > 0 else 30,
            processing_time=processing_time
        )
        
        # Bilgi panelini görüntüye ekle
        final_display = cv2.vconcat([labeled_frame, camera_info_panel])
        
        # Görüntüyü göster
        cv2.imshow(window_name, final_display)
        
        # 'q' tuşuna basınca çıkış
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def create_camera_info_panel(width, height, predictions, fps, processing_time):
    """Kamera için bilgi paneli oluştur"""
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (50, 50, 50)  # Koyu gri arkaplan
    
    # Üst çizgi
    cv2.line(panel, (0, 0), (width, 0), (100, 100, 100), 2)
    
    # Sol panel - bilgi
    info_text = f"FPS: {fps:.1f} | İşlem Süresi: {processing_time*1000:.1f}ms"
    cv2.putText(panel, info_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    # Sağ panel - tanınan yüzler
    face_text = f"Tanınan Yüzler: {len(predictions)}"
    cv2.putText(panel, face_text, (width - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    # Tanınan kişilerin listesi
    if predictions:
        unique_names = set()
        for _, _, _, _, name in predictions:
            if name != "Tanımlanamayan":
                unique_names.add(name)
        
        names_text = "Kişiler: " + ", ".join(unique_names) if unique_names else "Kişiler: -"
        cv2.putText(panel, names_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
    
    # Kontrol bilgisi
    cv2.putText(panel, "Çıkış: Q tuşu", (width - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
    
    return panel


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
    
    # Görüntüyü işle - daha iyi tanıma için
    print("[+] Yüzler tanınıyor...")
    start_time = time.time()
    preds = recognize_frame(img, known_encodings, known_names, detection_scale=0.5)
    process_time = time.time() - start_time
    
    if preds:
        print(f"[+] {len(preds)} yüz bulundu ({process_time:.2f} saniyede)")
        for i, (_, _, _, _, name) in enumerate(preds):
            print(f"    - Yüz {i+1}: {name}")
    else:
        print("[!] Fotoğrafta yüz bulunamadı.")
    
    # Etiketleri çiz
    labeled_img = draw_labels(img, preds)
    
    # Sonucu kaydet
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"out_{os.path.basename(img_path)}")
    cv2.imwrite(out_path, labeled_img)
    print(f"[+] İşlenmiş fotoğraf kaydedildi: {out_path}")
    
    # Görüntü çok büyükse küçült
    max_display_w = 1280
    max_display_h = 720
    display_img = labeled_img.copy()
    
    if display_img.shape[1] > max_display_w or display_img.shape[0] > max_display_h:
        scale = min(max_display_w / display_img.shape[1], max_display_h / display_img.shape[0])
        display_img = cv2.resize(labeled_img, (int(display_img.shape[1] * scale), int(display_img.shape[0] * scale)))
    
    # Bilgi paneli ekle
    info_height = 100
    info_panel = np.zeros((info_height, display_img.shape[1], 3), dtype=np.uint8)
    info_panel[:] = (50, 50, 50)  # Koyu gri
    
    # Bilgi metni
    cv2.putText(info_panel, f"Bulunan Yüzler: {len(preds)}", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(info_panel, f"İşlem Süresi: {process_time*1000:.1f}ms", (20, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    # Kişi isimleri
    if preds:
        unique_names = set()
        for _, _, _, _, name in preds:
            if name != "Tanımlanamayan":
                unique_names.add(name)
        
        names_text = "Kişiler: " + ", ".join(unique_names) if unique_names else "Tanınan kişi yok"
        cv2.putText(info_panel, names_text, (display_img.shape[1] - 400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 1)
    
    # Bilgi panelini ekle
    final_display = cv2.vconcat([display_img, info_panel])
    
    # Görüntüyü göster
    window_name = "Fotoğraf - Yüz Tanıma"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    cv2.imshow(window_name, final_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    print("\n===== YÜZ TANIMA SİSTEMİ =====")
    
    try:
        print("[+] Encodingler yükleniyor...")
        known_encodings, known_names = load_encodings()
        
        # Benzersiz kişileri say
        unique_names = set(known_names)
        print(f"[+] {len(known_names)} yüz yüklendi ({len(unique_names)} kişi)")
        
        for name in unique_names:
            count = known_names.count(name)
            print(f"    - {name}: {count} yüz")
            
        if len(known_names) == 0:
            print("[!] Hiç yüz yüklenmedi. Önce 'save_encodings.py' çalıştırın.")
            return
            
    except FileNotFoundError:
        print("[!] encodings.pkl dosyası bulunamadı. Önce 'save_encodings.py' çalıştırın.")
        return
    except Exception as e:
        print(f"[!] Encodingler yüklenirken hata: {e}")
        return
    
    while True:
        print("\nYapılacak işlemi seç:")
        print("1) Video işle")
        print("2) Kamera ile canlı tanıma")
        print("3) Fotoğraf işle")
        print("0) Çıkış")
        
        choice = input("\nSeçim (0/1/2/3): ").strip()
        
        if choice == "1":
            process_video(known_encodings, known_names)
        elif choice == "2":
            process_camera(known_encodings, known_names)
        elif choice == "3":
            process_image(known_encodings, known_names)
        elif choice == "0":
            print("[+] Program sonlandırılıyor...")
            break
        else:
            print("[!] Geçersiz seçim.")

if __name__ == "__main__":
    main()
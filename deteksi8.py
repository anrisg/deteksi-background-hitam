import cv2  # type: ignore
import time
import torch
import torchvision
from ultralytics import YOLO  # type: ignore

# Load model YOLOv8 hasil training
model_path = r"D:\yov8\best.pt"  # Ubah ke 'last.pt' jika diperlukan
model = YOLO(model_path)

# Inisialisasi kamera dengan backend DirectShow untuk mempercepat akses (Windows)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Cek apakah kamera berhasil terbuka
if not cap.isOpened():
    print("Tidak dapat membuka kamera laptop.")
    exit()

# Atur resolusi kamera ke 640x480 agar lebih cepat
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Variabel untuk menghitung FPS
prev_time = time.time()

# Loop untuk membaca frame dari kamera
while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari kamera.")
        break

    # Hitung FPS berdasarkan waktu antar frame
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    # Deteksi objek menggunakan model hasil training dengan threshold confidence 0.7
    results = model.predict(frame, conf=0.7)

    # Salin frame untuk anotasi
    annotated_frame = frame.copy()

    # Loop melalui setiap deteksi dan tambahkan bounding box serta nama kelas dengan confidence score
    for result in results:
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                x1, y1, x2, y2 = map(int, box)  # Koordinat bounding box
                class_id = int(cls)  # ID kelas
                confidence = float(conf)  # Confidence score

                # Ambil nama kelas berdasarkan model hasil training
                class_name = model.names.get(class_id, "Unknown")

                # Format label dengan nama kelas dan confidence score
                label = f"{class_name} ({confidence * 100:.2f}%)"

                # Gambar bounding box dan label pada frame
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    # Tambahkan FPS di tampilan
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Tampilkan hasil deteksi
    cv2.imshow('Deteksi Sampah dengan Model Custom', annotated_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan kamera dan tutup semua jendela
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)

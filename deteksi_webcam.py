import cv2
import numpy as np

# --- 1. Muat Model Pre-trained (SAMA SEPERTI SEBELUMNYA) ---
prototxt_path = "deploy.prototxt"
model_path = "res10_300x300_ssd_iter_140000.caffemodel"
model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
print("[INFO] Model sudah dimuat...")

# --- 2. Mulai Video Stream dari Webcam ---
print("[INFO] Memulai video stream dari webcam...")
# '0' berarti menggunakan webcam default. 
# Jika Anda punya >1 webcam, Anda bisa coba '1' atau '2'
cap = cv2.VideoCapture(0)

# --- 3. Loop Real-time ---
# Kita akan loop tanpa henti sampai Anda menekan tombol 'q'
while True:
    # Baca satu frame (satu gambar) dari video stream
    ret, frame = cap.read()

    # Jika frame tidak bisa dibaca (mis. webcam error), keluar dari loop
    if not ret:
        print("[ERROR] Gagal membaca frame dari webcam.")
        break

    # Dapatkan tinggi dan lebar frame
    (tinggi, lebar) = frame.shape[:2]

    # --- 4. Buat BLOB dan Jalankan Deteksi (SAMA SEPERTI SEBELUMNYA) ---
    
    # Buat "blob" dari frame
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, 
                                 (300, 300), (104.0, 177.0, 123.0))

    # Masukkan blob ke model dan dapatkan hasil deteksi
    model.setInput(blob)
    deteksi = model.forward()

    # --- 5. Loop Hasil Deteksi (SAMA SEPERTI SEBELUMNYA) ---
    for i in range(0, deteksi.shape[2]):
        # Dapatkan nilai "confidence"
        confidence = deteksi[0, 0, i, 2]

        # Filter deteksi yang lemah (di bawah 50%)
        if confidence > 0.5:
            # Dapatkan koordinat (x, y) untuk kotak
            box = deteksi[0, 0, i, 3:7] * np.array([lebar, tinggi, lebar, tinggi])
            (startX, startY, endX, endY) = box.astype("int")

            # Siapkan teks
            teks = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10

            # Gambar kotak dan teks PADA FRAME video
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)
            cv2.putText(frame, teks, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    # --- 6. Tampilkan Hasil (Real-time) ---
    # Tampilkan frame yang sudah digambari kotak
    cv2.imshow("Deteksi Wajah Real-Time (Tekan 'q' untuk keluar)", frame)
    
    # Cek apakah tombol 'q' (quit) ditekan
    # cv2.waitKey(1) akan menunggu 1 milidetik
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 7. Bersihkan ---
print("[INFO] Membersihkan dan menutup...")
cap.release() # Lepaskan webcam
cv2.destroyAllWindows() # Tutup semua jendela
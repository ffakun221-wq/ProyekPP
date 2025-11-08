import cv2
import os
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# --- 1. Muat SEMUA Model dan Data ---

# Model 1: Detektor Wajah (Caffe)
print("[INFO] Memuat model detektor wajah...")
prototxt_path = "deploy.prototxt"
model_path = "res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Model 2: Embedder Wajah (OpenFace)
print("[INFO] Memuat model embedder wajah...")
embedder_path = "openface.nn4.small2.v1.t7"
embedder = cv2.dnn.readNetFromTorch(embedder_path)

# Data 3: 'Database' Sidik Jari dan Label
print("[INFO] Memuat 'database' sidik jari wajah...")
embeddings_file = "embeddings.pickle"
data = pickle.loads(open(embeddings_file, "rb").read())

# --- 2. Siapkan dan Latih Classifier (SVM) ---
# Ini adalah model Machine Learning yang akan mencocokkan
# 'sidik jari' baru dengan 'sidik jari' di database

print("[INFO] Melatih model classifier (SVM)...")
# 'data["names"]' berisi nama-nama (mis: "Andi", "Andi", "Budi")
# 'LabelEncoder' mengubahnya jadi angka (mis: "Andi"=0, "Budi"=1)
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# 'data["embeddings"]' adalah list 'sidik jari' (vektor 128-d)
# Kita latih SVM (Support Vector Machine) untuk mengenali
# 'sidik jari' mana milik 'label' mana.
# [cite_start]Ini persis seperti yang dilakukan di jurnal [cite: 175]
recognizer = SVC(C=1.0, kernel="linear", probability=True) # [cite: 190]
recognizer.fit(data["embeddings"], labels)

# --- 3. Mulai Video Stream Webcam ---
print("[INFO] Memulai video stream dari webcam...")
cap = cv2.VideoCapture(0)

# --- 4. Loop Real-time ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]
    
    # --- LANGKAH A: DETEKSI Wajah (dari frame video) ---
    blob_detector = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                    (300, 300), (104.0, 177.0, 123.0))
    detector.setInput(blob_detector)
    detections = detector.forward()

    # Loop semua wajah yang terdeteksi
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5: # Filter deteksi lemah
            # --- LANGKAH B: CROP Wajah ---
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20: # Hindari wajah yg terlalu kecil
                continue

            # --- LANGKAH C: BUAT 'Sidik Jari' (dari wajah di video) ---
            blob_embedder = cv2.dnn.blobFromImage(face, 1.0 / 255.0, (96, 96),
                                            (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(blob_embedder)
            vec = embedder.forward() # 'Sidik jari' real-time

            # --- LANGKAH D: KENALI Wajah (Prediksi) ---
            # Gunakan classifier SVM untuk memprediksi
            # [cite: 269]
            preds = recognizer.predict_proba(vec)[0] 
            j = np.argmax(preds) # Ambil indeks probabilitas tertinggi
            proba = preds[j]     # Ambil nilai probabilitasnya
            name = le.classes_[j] # Dapatkan nama dari indeks tsb

            # --- LANGKAH E: Gambar Hasil ---
            # Siapkan teks: Nama + Persentase Keyakinan
            teks = f"{name}: {proba * 100:.2f}%"
            y = startY - 10 if startY - 10 > 10 else startY + 10

            # Gambar kotak dan teks
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)
            cv2.putText(frame, teks, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    # Tampilkan frame hasil
    cv2.imshow("Pengenalan Wajah (Tekan 'q' untuk keluar)", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 5. Bersihkan ---
print("[INFO] Membersihkan...")
cap.release()
cv2.destroyAllWindows()
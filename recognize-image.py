import cv2
import pickle
import numpy as np
import argparse  # Library untuk mengambil argumen dari terminal
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# --- 1. Parsing Argumen Terminal ---
# Kita buat cara agar bisa menjalankan: python recognize_image.py --image gambar_tes.jpg
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path ke gambar yang akan diuji")
args = vars(ap.parse_args())

# --- 2. Muat SEMUA Model dan Data ---
# (Ini semua sama persis dengan skrip webcam)

# Model 1: Detektor Wajah (Caffe)
prototxt_path = "deploy.prototxt"
model_path = "res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Model 2: Embedder Wajah (OpenFace)
embedder_path = "openface.nn4.small2.v1.t7"
embedder = cv2.dnn.readNetFromTorch(embedder_path)

# Data 3: 'Database' Sidik Jari dan Label
embeddings_file = "embeddings.pickle"
data = pickle.loads(open(embeddings_file, "rb").read())

# --- 3. Siapkan dan Latih Classifier (SVM) ---
# (Sama persis dengan skrip webcam)

le = LabelEncoder()
labels = le.fit_transform(data["names"])

recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# --- 4. Muat dan Proses Gambar Input ---
# BEDA: Bukan lagi 'cap.read()', tapi 'cv2.imread()'
print(f"[INFO] Memproses gambar: {args['image']}...")
frame = cv2.imread(args["image"]) # Memuat gambar dari file

if frame is None:
    print(f"[ERROR] Gagal memuat gambar. Periksa path: {args['image']}")
    exit()

(h, w) = frame.shape[:2]

# --- 5. Jalankan Deteksi dan Pengenalan ---
# (Logik ini sama, tapi tidak di dalam loop 'while')

# Buat blob untuk detektor
blob_detector = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                (300, 300), (104.0, 177.0, 123.0))
detector.setInput(blob_detector)
detections = detector.forward()

# Loop semua wajah yang terdeteksi di gambar
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    if confidence > 0.5: # Filter deteksi lemah
        # --- CROP Wajah ---
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        face = frame[startY:endY, startX:endX]
        (fH, fW) = face.shape[:2]

        if fW < 20 or fH < 20:
            continue

        # --- BUAT 'Sidik Jari' ---
        blob_embedder = cv2.dnn.blobFromImage(face, 1.0 / 255.0, (96, 96),
                                        (0, 0, 0), swapRB=True, crop=False)
        embedder.setInput(blob_embedder)
        vec = embedder.forward()

        # --- KENALI Wajah (Prediksi) ---
        preds = recognizer.predict_proba(vec)[0]
        j = np.argmax(preds)
        proba = preds[j]
        name = le.classes_[j]

        # --- Gambar Hasil ---
        teks = f"{name}: {proba * 100:.2f}%"
        y = startY - 10 if startY - 10 > 10 else startY + 10
        
        cv2.rectangle(frame, (startX, startY), (endX, endY),
                      (0, 255, 0), 2)
        cv2.putText(frame, teks, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

# --- 6. Tampilkan Hasil ---
# BEDA: Kita tampilkan sekali di akhir, lalu tunggu tombol ditekan
cv2.imshow("Hasil Pengenalan Wajah", frame)
cv2.waitKey(0) # Tunggu sampai tombol ditekan
cv2.destroyAllWindows()
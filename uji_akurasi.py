import cv2
import numpy as np
import pickle
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import time

# --- 1. Muat Model Embedder ---
print("[INFO] Memuat model embedder wajah (OpenFace)...")
embedder_path = "openface.nn4.small2.v1.t7"
embedder = cv2.dnn.readNetFromTorch(embedder_path)

# --- 2. Muat dan Bagi Dataset Olivetti ---
print("[INFO] Memuat dan membagi dataset Olivetti...")
dataset = fetch_olivetti_faces()
images = dataset.images
labels = dataset.target

# BAGI DATA: 80% Latih, 20% Uji

X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.20, stratify=labels, random_state=42)

print(f"[INFO] Data Latih: {len(X_train)} gambar")
print(f"[INFO] Data Uji: {len(X_test)} gambar")

# --- 3. FASE LATIH (Encode Data Latih) ---
print("[INFO] Memproses Data Latih untuk membuat 'sidik jari'...")
knownEmbeddings = []
knownNames = []

start_time = time.time()
# Loop pada semua Data Latih
for i in range(len(X_train)):
    image_float = X_train[i]
    label = y_train[i]

    
    image_uint8 = (image_float * 255).astype(np.uint8)
    face = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)

    # Buat 'sidik jari'
    blob_embedder = cv2.dnn.blobFromImage(face, 1.0 / 255.0, (96, 96),
                                    (0, 0, 0), swapRB=True, crop=False)
    embedder.setInput(blob_embedder)
    vec = embedder.forward()
    
    knownNames.append(str(label))
    knownEmbeddings.append(vec.flatten())

print(f"[INFO] Selesai encode Data Latih dalam {time.time() - start_time:.2f} detik.")

# --- 4. FASE LATIH (Train Classifier SVM) ---
print("[INFO] Melatih model classifier (SVM)...")
le = LabelEncoder()
labels_encoded = le.fit_transform(knownNames)

recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(knownEmbeddings, labels_encoded)

# --- 5. FASE UJI (Evaluasi pada Data Uji) ---
print("[INFO] Mengevaluasi model pada Data Uji...")
correct_predictions = 0

# Loop pada semua Data Uji
for i in range(len(X_test)):
    image_float = X_test[i]
    true_label = str(y_test[i])

    
    image_uint8 = (image_float * 255).astype(np.uint8)
    face = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)

    # Buat 'sidik jari'
    blob_embedder = cv2.dnn.blobFromImage(face, 1.0 / 255.0, (96, 96),
                                    (0, 0, 0), swapRB=True, crop=False)
    embedder.setInput(blob_embedder)
    vec = embedder.forward()

    # Prediksi
    preds = recognizer.predict_proba(vec)[0]
    j = np.argmax(preds)
    predicted_label = le.classes_[j] 

    # Bandingkan
    if predicted_label == true_label:
        correct_predictions += 1
        print(f"  > Tes {i+1}/{len(X_test)}: Menebak: {predicted_label} | Sebenarnya: {true_label} (BENAR)")
    else:
        print(f"  > Tes {i+1}/{len(X_test)}: Menebak: {predicted_label} | Sebenarnya: {true_label} (SALAH)")

# --- 6. Tampilkan Hasil Akhir ---
accuracy = (correct_predictions / len(X_test)) * 100
print("\n" + "="*30)
print("[INFO] HASIL ANALISIS SELESAI")
print(f"Total Gambar Uji    : {len(X_test)}")
print(f"Total Tebakan Benar : {correct_predictions}")
print(f"Total Tebakan Salah : {len(X_test) - correct_predictions}")
print(f"AKURASI FINAL       : {accuracy:.2f}%")
print("="*30)
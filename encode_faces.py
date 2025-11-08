import cv2
import os
import numpy as np
import pickle # Kita akan pakai ini untuk menyimpan hasil 'sidik jari'

# --- 1. Muat SEMUA Model ---

# Model 1: Detektor Wajah (Caffe) - Untuk MENEMUKAN wajah
print("[INFO] Memuat model detektor wajah...")
prototxt_path = "deploy.prototxt"
model_path = "res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Model 2: Embedder Wajah (OpenFace) - Untuk MEMBUAT 'sidik jari'
print("[INFO] Memuat model embedder wajah...")
embedder_path = "openface.nn4.small2.v1.t7"
embedder = cv2.dnn.readNetFromTorch(embedder_path)

# --- 2. Tentukan Path dan Siapkan Variabel ---

dataset_folder = "lfw-deepfunneled" # Nama folder dataset Anda
embeddings_file = "embeddings.pickle" # Nama file output 'database' kita

# Siapkan list kosong untuk menampung 'sidik jari' dan namanya
knownEmbeddings = []
knownNames = []

total_faces_processed = 0

print("[INFO] Memproses gambar di folder dataset...")

# --- 3. Loop Folder Dataset ---

# Loop semua folder orang (misal: 'Andi', 'Budi', dll)
for person_name in os.listdir(dataset_folder):
    person_folder_path = os.path.join(dataset_folder, person_name)

    # Pastikan itu adalah folder
    if not os.path.isdir(person_folder_path):
        continue

    # Loop semua gambar di dalam folder orang tsb
    for image_name in os.listdir(person_folder_path):
        image_path = os.path.join(person_folder_path, image_name)

        # Muat gambar
        image = cv2.imread(image_path)
        if image is None:
            print(f"[WARNING] Gagal memuat gambar: {image_path}")
            continue
            
        (h, w) = image.shape[:2]

        # --- LANGKAH A: DETEKSI Wajah (Sama seperti skrip deteksi) ---
        
        # Buat blob untuk detektor
        blob_detector = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                        (300, 300), (104.0, 177.0, 123.0))

        detector.setInput(blob_detector)
        detections = detector.forward()

        # Kita asumsikan hanya ada SATU wajah per gambar dataset
        # Ambil deteksi dengan confidence tertinggi
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        # Filter confidence
        if confidence > 0.5: # Anda bisa naikkan threshold jika perlu
            # Dapatkan kotak wajah
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Crop wajah dari gambar
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # Pastikan wajah cukup besar
            if fW < 20 or fH < 20:
                print(f"[WARNING] Wajah terlalu kecil di: {image_path}")
                continue

            # --- LANGKAH B: EKSTRAKSI Fitur ('Sidik Jari') ---
            
            # Buat blob untuk embedder (model OpenFace)
            # Model ini butuh input 96x96 piksel
            blob_embedder = cv2.dnn.blobFromImage(face, 1.0 / 255.0, (96, 96),
                                            (0, 0, 0), swapRB=True, crop=False)
            
            embedder.setInput(blob_embedder)
            vec = embedder.forward() # Ini adalah 'sidik jari' 128-d
            
            # Tambahkan nama dan 'sidik jari' ke list kita
            knownNames.append(person_name)
            knownEmbeddings.append(vec.flatten()) # flatten() untuk mengubahnya jadi 1 baris
            total_faces_processed += 1
        
        else:
            print(f"[WARNING] Tidak ada wajah terdeteksi di: {image_path}")

# --- 4. Simpan 'Database' Sidik Jari ---

print(f"[INFO] Selesai. Total {total_faces_processed} wajah di-encode.")

# Gabungkan data jadi satu
data = {"embeddings": knownEmbeddings, "names": knownNames}

# Simpan ke file 'pickle'
with open(embeddings_file, "wb") as f:
    f.write(pickle.dumps(data))

print(f"[INFO] 'Sidik jari' wajah disimpan ke {embeddings_file}")
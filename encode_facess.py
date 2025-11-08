import cv2
import os
import numpy as np
import pickle
from sklearn.datasets import fetch_olivetti_faces # <-- BARU

# --- 1. Muat Model Embedder (Kita TIDAK PERLU Caffe Detector) ---

# Model 2: Embedder Wajah (OpenFace) - Untuk MEMBUAT 'sidik jari'
print("[INFO] Memuat model embedder wajah...")
embedder_path = "openface.nn4.small2.v1.t7"
embedder = cv2.dnn.readNetFromTorch(embedder_path)

# --- 2. Muat Dataset Olivetti ---
print("[INFO] Memuat dataset Olivetti (ORL)...")
# Ini akan mengunduh dan memuat datasetnya
# Dataset ini berisi 400 gambar (40 orang, @ 10 gambar)
orl_dataset = fetch_olivetti_faces()

# images: data gambar (float 0.0-1.0, ukuran 64x64)
# target: label/nama (angka 0-39)
images = orl_dataset.images
labels = orl_dataset.target

# --- 3. Siapkan Variabel ---
embeddings_file = "embeddings.pickle" # Nama file output 'database'

knownEmbeddings = []
knownNames = []
total_faces_processed = 0

print("[INFO] Memproses gambar di dataset Olivetti...")

# --- 4. Loop Dataset Olivetti ---
# Kita loop semua 400 gambar
for i in range(len(images)):
    image_float = images[i]  # Ini gambar 64x64 (grayscale, float 0-1)
    label = labels[i]        # Ini labelnya (mis: 0, 1, ... 39)

    # --- LANGKAH A: Konversi Gambar ---
    # Model OpenFace butuh gambar BGR/RGB (uint8)
    
    # 1. Ubah float 0-1 menjadi uint8 0-255
    image_uint8 = (image_float * 255).astype(np.uint8)
    
    # 2. Ubah Grayscale (1 channel) menjadi RGB (3 channel)
    # Kita duplikat channel abu-abu ke 3 channel
    face = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)
    
    # --- LANGKAH B: EKSTRAKSI Fitur ('Sidik Jari') ---
    # (Kita lewati deteksi karena gambar sudah di-crop)
    
    # Buat blob untuk embedder (model OpenFace)
    # Model ini butuh input 96x96 piksel, 
    # blobFromImage akan me-resize 64x64 kita ke 96x96
    blob_embedder = cv2.dnn.blobFromImage(face, 1.0 / 255.0, (96, 96),
                                    (0, 0, 0), swapRB=True, crop=False)
    
    embedder.setInput(blob_embedder)
    vec = embedder.forward() # Ini adalah 'sidik jari' 128-d
    
    # Tambahkan nama (label) dan 'sidik jari' ke list kita
    # Kita ubah label angka (mis: 39) jadi string ("39")
    knownNames.append(str(label))
    knownEmbeddings.append(vec.flatten())
    total_faces_processed += 1

# --- 5. Simpan 'Database' Sidik Jari ---

print(f"[INFO] Selesai. Total {total_faces_processed} wajah di-encode.")

# Gabungkan data jadi satu
data = {"embeddings": knownEmbeddings, "names": knownNames}

# Simpan ke file 'pickle'
with open(embeddings_file, "wb") as f:
    f.write(pickle.dumps(data))

print(f"[INFO] 'Sidik jari' wajah disimpan ke {embeddings_file}")
import matplotlib.pyplot as plt # Library utama untuk plotting (gambar grafik/foto)
from sklearn.datasets import fetch_olivetti_faces
import numpy as np
import time

print("[INFO] Memuat dataset Olivetti...")
dataset = fetch_olivetti_faces()

# Kita ambil data gambar dan labelnya
images = dataset.images
labels = dataset.target

print(f"[INFO] Total gambar: {len(images)}")
print(f"[INFO] Total orang (label unik): {len(np.unique(labels))}")
print("[INFO] Menampilkan 25 contoh gambar pertama...")
time.sleep(1) # Beri jeda 1 detik

# --- Mari kita tampilkan 25 gambar pertama ---

# Kita buat 'kanvas' (figure) 5 baris x 5 kolom
# figsize menentukan ukuran jendela
fig, ax = plt.subplots(20, 20, figsize=(25, 25))

for i in range(len(images)):
    # Hitung posisi baris dan kolom di grid
    # Misal i=0 -> baris=0, kolom=0
    # Misal i=6 -> baris=1, kolom=1
    baris = i // 20
    kolom = i % 20
    
    # Ambil gambar ke-i dan labelnya
    gambar = images[i]
    label = labels[i]
    
    # Tampilkan gambar di subplot (posisi baris, kolom)
    # 'cmap='gray'' SANGAT PENTING karena ini gambar hitam-putih
    ax[baris, kolom].imshow(gambar, cmap='gray')
    
    # Tampilkan labelnya (0-39) sebagai judul
    ax[baris, kolom].set_title(f"Label (Orang): {label}")
    
    # Sembunyikan sumbu x/y (angka-angka 0-64) agar bersih
    ax[baris, kolom].axis('off') 

# Beri judul utama
plt.suptitle("Contoh 400 Gambar Pertama dari Dataset Olivetti", fontsize=16)

# Rapikan layout
plt.tight_layout(rect=[0, 0.03, 1, 0.97])

# Tampilkan jendelanya!
plt.show()
import cv2
import numpy as np

prototxt_path = 'deploy.prototxt'
model_path = 'res10_300x300_ssd_iter_140000.caffemodel'

print("[INFO] Memuat model...")
model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)


nama_gambar = "gambar_wajah.jpg"
gambar = cv2.imread(nama_gambar)

(tinggi, lebar) = gambar.shape[:2]

blob = cv2.dnn.blobFromImage(cv2.resize(gambar, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

print("[INFO] Melakukan Deteksi Wajah...")

model.setInput(blob)

deteksi = model.forward()


for i in range(0, deteksi.shape[2]):
    confidience = deteksi[0, 0, i, 2]

    if confidience > 0.5:
        box = deteksi[0, 0, i, 3:7] * np.array([lebar, tinggi, lebar, tinggi])
        (startX, startY, endX, endY) = box.astype("int")

        teks = "{:.2f}%".format(confidience * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10


        cv2.rectangle(gambar, (startX, startY), (endX, endY), (0, 255, 0), 2)

        cv2.putText(gambar, teks, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)


cv2.imshow("Hasil Deteksi Wajah", gambar)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Deteksi-Gambar-YOLOv8
Nama: Habibatul Khoiroh
NIM: 23422036

```python
# Step 1: Install necessary libraries
!pip install ultralytics  # Install YOLOv8
!pip install matplotlib opencv-python-headless
!pip install roboflow

# Step 2: Import libraries
import matplotlib.pyplot as plt
from ultralytics import YOLO
from google.colab import files
import cv2
import numpy as np

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="LuNXCgqzh3opWhWacuBv")
project = rf.workspace("deteksi-kdtt3").project("scissors-2wucz")
version = project.version(1)
dataset = version.download("yolov8")

import os

# Lihat folder tempat dataset diunduh
dataset_location = dataset.location  # dari RoboFlow download
print("Dataset downloaded to:", dataset_location)

from ultralytics import YOLO

# Buat model YOLOv8 baru
model = YOLO("yolov8n.pt")  # "yolov8n.pt" adalah versi YOLOv8 Nano

# Jalankan pelatihan dengan dataset
model.train(data="/content/scissors-1/data.yaml", epochs=25, imgsz=640)

dataset = version.download("yolov8")

result = model.predict(source="/content/0f89f6766ba51a7901f0a58c555a5092.jpg", save=True, imgsz=640)

# Ambil elemen pertama dari hasil prediksi
image_result = result[0]

# Menampilkan hasil deteksi
from IPython.display import Image, display
image_path_with_predictions = image_result.plot()  # Mengembalikan array gambar

# Tampilkan menggunakan Matplotlib
import matplotlib.pyplot as plt
plt.imshow(image_path_with_predictions)
plt.axis("off")
plt.show()
```

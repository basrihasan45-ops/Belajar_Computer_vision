from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load model YOLO (pretrained)
model = YOLO('yolov8n.pt')  # model ringan

# Load gambar
image_path = 'objekPred.jpeg'  # ganti dengan path gambar kamu
img = cv2.imread(image_path)

# Deteksi objek
results = model(img)

# Visualisasi hasil
annotated_img = results[0].plot()

# Tampilkan hasil
plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Print hasil deteksi
for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls]
        print(f"Objek: {label}, Confidence: {conf:.2f}")

from ultralytics import YOLO
import cv2

app_path = "/home/deist/Downloads/OCR/anh/"
model = YOLO(app_path + "yolov8n-electronic-bbox.pt")

img = cv2.imread(app_path + "yolo_electronic_dataset/images/train/2aoboqvcsbtbq1l25bfe6rmdrmn3uwnaqrpjwfuk1060.jpg")
results = model(img, conf=0.1)
print(results[0].boxes)

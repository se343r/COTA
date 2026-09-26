from ultralytics import YOLO
import cv2

model = YOLO("/home/deist/Downloads/OCR/anh/yolov8n-digits-bbox.pt")
img = cv2.imread("/home/deist/Downloads/OCR/anh/filtered/2aoboqvcsbtbq1l25bfe6rmdrmn3uwnaqrpjwfuk1060.jpg")
res = model(img)
for box in res[0].boxes:
    print(f"Class: {box.cls[0].item()}, conf: {box.conf[0].item()}, xyxy: {box.xyxy[0].tolist()}")

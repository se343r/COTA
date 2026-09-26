from ultralytics import YOLO
import cv2

model = YOLO("/home/deist/Downloads/OCR/anh/yolov8n-electronic-bbox.pt")
print(model.names)

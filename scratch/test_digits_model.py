from ultralytics import YOLO
model = YOLO("/home/deist/Downloads/OCR/anh/yolov8n-digits.pt")
print(model.names)

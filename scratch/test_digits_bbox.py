from ultralytics import YOLO
model = YOLO("/home/deist/Downloads/OCR/anh/yolov8n-digits-bbox.pt")
print(model.names)

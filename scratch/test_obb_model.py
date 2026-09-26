from ultralytics import YOLO
model = YOLO("/home/deist/Downloads/OCR/anh/yolov8n-obb.pt")
print(model.names)

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

img_path = "/home/deist/Downloads/OCR/testocr/2aoboqyumogfwvc9fiskxwswtdkljso9hh2he6i02.jpg"
img = cv2.imread(img_path)

obb_model = YOLO("yolov8n-obb.pt")
r = obb_model(img, verbose=False, conf=0.25)[0]
obb = r.obb
print(f"Found {len(obb)} OBBs")
for i in range(len(obb)):
    conf = float(obb.conf[i])
    cls = int(obb.cls[i])
    quad = [tuple(p) for p in obb.xyxyxyxy[i].tolist()]
    pts = np.array(quad, dtype=np.float32)
    w1 = int(np.linalg.norm(pts[0] - pts[1]))
    w2 = int(np.linalg.norm(pts[2] - pts[3]))
    h1 = int(np.linalg.norm(pts[1] - pts[2]))
    h2 = int(np.linalg.norm(pts[0] - pts[3]))
    print(f"OBB {i}: cls={cls} conf={conf:.3f} w={max(w1, w2)} h={max(h1, h2)}")

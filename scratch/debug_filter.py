import cv2
import numpy as np
from ultralytics import YOLO

img_path = "/home/deist/Downloads/OCR/testocr/2aoboqyunmlub0obsmvj8tze1a3ai1boim53x4ua32.jpg"
img = cv2.imread(img_path)

obb_model = YOLO("runs/obb/train/weights/best.pt")
r = obb_model(img, verbose=False, conf=0.25)[0]
obb = r.obb

best_idx = np.argmax(obb.conf.cpu().numpy())
quad = [tuple(p) for p in obb.xyxyxyxy[best_idx].tolist()]

pts_sum = [p[0] + p[1] for p in quad]
pts_diff = [p[0] - p[1] for p in quad]
tl = quad[np.argmin(pts_sum)]
br = quad[np.argmax(pts_sum)]
tr = quad[np.argmax(pts_diff)]
bl = quad[np.argmin(pts_diff)]
quad = [tl, tr, br, bl]

pts = np.array(quad, dtype=np.float32)
pad = 20
w1 = int(np.linalg.norm(pts[0] - pts[1]))
w2 = int(np.linalg.norm(pts[2] - pts[3]))
h1 = int(np.linalg.norm(pts[1] - pts[2]))
h2 = int(np.linalg.norm(pts[0] - pts[3]))
max_w = max(w1, w2)
max_h = max(h1, h2)

dst = np.array([
    [pad, pad],
    [pad + max_w, pad],
    [pad + max_w, pad + max_h],
    [pad, pad + max_h]
], dtype=np.float32)

M = cv2.getPerspectiveTransform(pts, dst)
warped = cv2.warpPerspective(img, M, (max_w + 2*pad, max_h + 2*pad))

model_bbox = YOLO("yolov8n-digits-bbox.pt")
res = model_bbox(warped, conf=0.35, iou=0.45, verbose=False)[0]

print(f"Max W: {max_w}, Max H: {max_h}, Pad: {pad}")
for bx in res.boxes:
    x1, y1, x2, y2 = map(float, bx.xyxy[0].tolist())
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    in_bounds = cx >= pad and cx <= pad + max_w and cy >= pad and cy <= pad + max_h
    print(f"Box: {x1:.1f}, {y1:.1f} -> {x2:.1f}, {y2:.1f} | Center: {cx:.1f}, {cy:.1f} | In bounds: {in_bounds}")


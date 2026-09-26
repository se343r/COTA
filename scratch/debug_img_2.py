import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.append("/home/deist/Downloads/OCR/anh")
from app import _load_model, get_yolo_electronic

obb_model = _load_model()
elec_model = get_yolo_electronic()

img_path = "/home/deist/Downloads/OCR/testocr/2aoboqyuoytaljd4yk4vonzulqxpesc2snwugnhe47.jpg"
img = cv2.imread(img_path)
r = obb_model(img, verbose=False, conf=0.25)[0]
obb = r.obb
best_idx = 0
quad = [tuple(p_) for p_ in obb.xyxyxyxy[best_idx].tolist()]
pts_sum = [p_[0] + p_[1] for p_ in quad]
pts_diff = [p_[0] - p_[1] for p_ in quad]
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
dst = np.array([[pad, pad], [pad + max_w, pad], [pad + max_w, pad + max_h], [pad, pad + max_h]], dtype=np.float32)
M = cv2.getPerspectiveTransform(pts, dst)
warped = cv2.warpPerspective(img, M, (max_w + 2*pad, max_h + 2*pad))

r_bbox = elec_model(warped, conf=0.1, verbose=False)[0]
print("Boxes at 0.1:", len(r_bbox.boxes))
for bx in r_bbox.boxes:
    print("Conf:", float(bx.conf[0]))
    
    cx = (float(bx.xyxy[0][0]) + float(bx.xyxy[0][2])) / 2
    cy = (float(bx.xyxy[0][1]) + float(bx.xyxy[0][3])) / 2
    print(f"Center: {cx}, {cy} | Bounds: pad={pad}, pad+max_w={pad+max_w}, pad+max_h={pad+max_h}")

import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.append("/home/deist/Downloads/OCR/anh")
from app import _load_model, get_yolo_electronic, get_yolo_digits

obb_model = _load_model()
elec_model = get_yolo_electronic()

for p in Path("/home/deist/Downloads/OCR/anh/test_images").glob("*.jpg"):
    img = cv2.imread(str(p))
    r = obb_model(img, verbose=False, conf=0.25)[0]
    obb = getattr(r, "obb", None)
    if obb is None or len(obb) == 0: continue
    
    best_idx, best_conf = 0, 0
    for i in range(len(obb)):
        c = float(obb.conf[i].item())
        if c > best_conf:
            best_conf = c
            best_idx = i
            
    cls_id = int(obb.cls[best_idx].item())
    if cls_id != 1: continue
    
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
    
    r_bbox = elec_model(warped, conf=0.2, verbose=False)[0]
    if len(r_bbox.boxes) == 0:
        print(f"FAILED on {p.name}")
    else:
        max_conf = max([float(bx.conf[0].item()) for bx in r_bbox.boxes])
        if max_conf < 0.5:
            print(f"LOW CONF {max_conf:.2f} on {p.name}")

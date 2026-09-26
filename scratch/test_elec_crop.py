import cv2
import numpy as np
import torch
from PIL import Image
import sys
sys.path.append("/home/deist/Downloads/OCR/anh")
from app import _load_model, get_yolo_electronic, get_parseq_model

img_path = "/home/deist/Downloads/OCR/testocr/2aoboqyuoytaljd4yk4vonzulqxpesc2snwugnhe47.jpg"
img = cv2.imread(img_path)

obb_model = _load_model()
elec_model = get_yolo_electronic()
parseq_model, parseq_transform = get_parseq_model()

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
cv2.imwrite("/home/deist/Downloads/OCR/anh/scratch/warped_test.jpg", warped)

r_bbox = elec_model(warped, conf=0.25, verbose=False)[0]
for i, bx in enumerate(r_bbox.boxes):
    x1, y1, x2, y2 = map(int, bx.xyxy[0].tolist())
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(warped.shape[1], x2); y2 = min(warped.shape[0], y2)
    crop = warped[y1:y2, x1:x2]
    cv2.imwrite(f"/home/deist/Downloads/OCR/anh/scratch/crop_{i}.jpg", crop)
    
    crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    inp = parseq_transform(crop_pil).unsqueeze(0).to("cuda")
    with torch.no_grad():
        logits = parseq_model(inp)
        pred = logits.softmax(-1)
        label_pred, prob = parseq_model.tokenizer.decode(pred)
        print(f"Crop {i}: {label_pred[0]}")

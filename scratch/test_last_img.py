import sys, cv2, json
import numpy as np
sys.path.append("/home/deist/Downloads/OCR/anh")
from app import get_yolo_digits, get_yolo_electronic

img = cv2.imread("/home/deist/Downloads/OCR/anh/filtered/2aoboqyunlb9lfvfwgylx3lrcmynnwkgcw5rkxcw36.jpg")
with open("/home/deist/Downloads/OCR/anh/crops/digits_data/2aoboqyunlb9lfvfwgylx3lrcmynnwkgcw5rkxcw36.json") as f:
    d = json.load(f)
pts = np.array(d['corners'], dtype=np.float32)
pad = 5
max_w = max(int(np.linalg.norm(pts[0]-pts[1])), int(np.linalg.norm(pts[2]-pts[3])))
max_h = max(int(np.linalg.norm(pts[1]-pts[2])), int(np.linalg.norm(pts[0]-pts[3])))
dst = np.array([[pad,pad],[pad+max_w,pad],[pad+max_w,pad+max_h],[pad,pad+max_h]], np.float32)
M = cv2.getPerspectiveTransform(pts, dst)
warped = cv2.warpPerspective(img, M, (max_w+2*pad, max_h+2*pad))

model_bbox = get_yolo_digits()
res = model_bbox(warped, conf=0.35, iou=0.45, verbose=False)
for bx in res[0].boxes:
    print(bx.cls[0].item(), bx.conf[0].item(), bx.xyxy[0].tolist())

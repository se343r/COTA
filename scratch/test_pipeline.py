import cv2, json, torch, numpy as np
from PIL import Image
from pathlib import Path

import sys
sys.path.append("/home/deist/Downloads/OCR/anh")
from app import get_parseq_model, get_yolo_electronic

def warp(img, corners):
    pts = np.array(corners, dtype=np.float32)
    pad = 5
    max_w = max(int(np.linalg.norm(pts[0]-pts[1])), int(np.linalg.norm(pts[2]-pts[3])))
    max_h = max(int(np.linalg.norm(pts[1]-pts[2])), int(np.linalg.norm(pts[0]-pts[3])))
    dst = np.array([[pad,pad],[pad+max_w,pad],[pad+max_w,pad+max_h],[pad,pad+max_h]], np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(img, M, (max_w+2*pad, max_h+2*pad)), M, max_w, max_h, pad

img_path = "/home/deist/Downloads/OCR/anh/filtered/2aoboqvcsbtbq1l25bfe6rmdrmn3uwnaqrpjwfuk1060.jpg"
json_path = "/home/deist/Downloads/OCR/anh/crops/digits_data/2aoboqvcsbtbq1l25bfe6rmdrmn3uwnaqrpjwfuk1060.json"

with open(json_path) as f: d = json.load(f)
img = cv2.imread(img_path)
warped, M, max_w, max_h, pad = warp(img, d['corners'])

# 1. PARSeq reads the whole screen
p_model, p_tf = get_parseq_model()
device = "cuda" if torch.cuda.is_available() else "cpu"
inp = p_tf(Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
with torch.no_grad():
    logits = p_model(inp)
    label, _ = p_model.tokenizer.decode(logits.softmax(-1))
    pred_str = label[0]

# 2. YOLO electronic finds the sequence bounding box
model_bbox = get_yolo_electronic()
res = model_bbox(warped, verbose=False)
if len(res[0].boxes) > 0:
    box = res[0].boxes[0]
    bx_x1, bx_y1, bx_x2, bx_y2 = map(float, box.xyxy[0].tolist())
else:
    bx_x1, bx_y1, bx_x2, bx_y2 = pad, pad, pad+max_w, pad+max_h

print(f"PARSeq predicted string: {pred_str}")
print(f"YOLO bbox: {bx_x1, bx_y1, bx_x2, bx_y2}")

# 3. Slice the YOLO bbox uniformly
n = len(pred_str)
if n == 0: n = 1
step = (bx_x2 - bx_x1) / n
for i in range(n):
    print(f"  char {pred_str[i] if i<len(pred_str) else '?'}: [{bx_x1 + i*step:.1f}, {bx_y1:.1f}, {bx_x1 + (i+1)*step:.1f}, {bx_y2:.1f}]")


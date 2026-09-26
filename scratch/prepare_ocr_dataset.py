import json
import shutil
from pathlib import Path
import cv2
import numpy as np

BASE_DIR = Path("/home/deist/Downloads/OCR/anh")
DIGITS_DIR = BASE_DIR / "crops" / "digits_data"
OCR_DIR = BASE_DIR / "ocr_dataset"

if OCR_DIR.exists():
    shutil.rmtree(OCR_DIR)

OCR_DIR.mkdir()
CROP_IMAGES_DIR = OCR_DIR / "crops"
LABELS_DIR = OCR_DIR / "labels"
CROP_IMAGES_DIR.mkdir()
LABELS_DIR.mkdir()

def find_image(name):
    for p in (BASE_DIR / "filtered").rglob(name):
        if p.is_file():
            return p
    for p in BASE_DIR.rglob(name):
        if "yolo" not in str(p) and "scratch" not in str(p):
            return p
    return None

json_files = list(DIGITS_DIR.glob("*.json"))
generated = 0

for jf in json_files:
    with open(jf, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    class_id = data.get("class_id", 0)
    # Only keep mechanical meters (class 0)
    if class_id != 0:
        continue
        
    digits = data.get("digits", [])
    if not digits:
        continue
        
    img_name = data["image"]
    img_path = find_image(img_name)
    if not img_path:
        continue
        
    img = cv2.imread(str(img_path))
    if img is None: continue
    
    pts = np.array(data["corners"], dtype=np.float32) 
    
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
    
    crop_id = jf.stem
    cv2.imwrite(str(CROP_IMAGES_DIR / f"{crop_id}.jpg"), warped)
    
    out_digits = []
    for i, d in enumerate(digits):
        cx, cy, w, h = d["cx"], d["cy"], d["w"], d["h"]
        x = cx - w / 2
        y = cy - h / 2
        out_digits.append({
            "digit_index": i,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "label": d.get("label", ""),
            "is_decimal": d.get("is_decimal", False),
            "mid_transition": d.get("is_between", False)
        })
        
    out_data = {
        "crop_id": crop_id,
        "source_image": img_name,
        "digits": out_digits
    }
    
    with open(LABELS_DIR / f"{crop_id}.json", "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
        
    generated += 1

print(f"Generated {generated} OCR items at {OCR_DIR}")

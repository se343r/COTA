import json
import shutil
from pathlib import Path
import cv2
import numpy as np

BASE_DIR = Path("/home/deist/Downloads/OCR/anh")
DIGITS_DIR = BASE_DIR / "crops" / "digits_data"

def find_image(name):
    for p in (BASE_DIR / "filtered").rglob(name):
        if p.is_file():
            return p
    for p in BASE_DIR.rglob(name):
        if "yolo" not in str(p):
            return p
    return None

YOLO_DIR = BASE_DIR / "yolo_digits_dataset"
if YOLO_DIR.exists():
    shutil.rmtree(YOLO_DIR)

YOLO_DIR.mkdir()
(YOLO_DIR / "images" / "train").mkdir(parents=True)
(YOLO_DIR / "images" / "val").mkdir(parents=True)
(YOLO_DIR / "labels" / "train").mkdir(parents=True)
(YOLO_DIR / "labels" / "val").mkdir(parents=True)

json_files = list(DIGITS_DIR.glob("*.json"))
np.random.seed(42)
np.random.shuffle(json_files)
split_idx = int(len(json_files) * 0.8)

generated = 0

for i, jp in enumerate(json_files):
    split = "train" if i < split_idx else "val"
    
    with open(jp, "r", encoding="utf-8") as f:
        data = json.load(f)
        
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
    
    out_img_path = YOLO_DIR / "images" / split / f"{jp.stem}.jpg"
    cv2.imwrite(str(out_img_path), warped)
    
    out_lbl_path = YOLO_DIR / "labels" / split / f"{jp.stem}.txt"
    lines = []
    
    cls_id = data.get("class_id", 0)
    digits = data.get("digits", [])
    if not digits:
        continue 
        
    for d in digits:
        # Bỏ qua các nửa số hoặc không đọc được
        if d.get("is_between", False):
            continue
        label = d.get("label", "")
        if label == "unreadable" or label == "" or label == "?":
            continue
            
        if cls_id == 0:
            # Maybe the user wants actual classes? "nhận diện thế nào là số nhé"
            # We can map 0-9 if they want classification. But they said "chỉ cần tạo bbox, còn nhãn là để train model OCR" previously.
            # I will stick to 1 class for digits, but only for clean digits!
            c = 0 # digit
        else:
            c = 1 # electronic_sequence
            
        cx, cy, w, h = d["cx"], d["cy"], d["w"], d["h"]
        lines.append(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        
    # Only save if there are still boxes left
    if lines:
        out_lbl_path.write_text("\n".join(lines))
        generated += 1
    else:
        # if no valid boxes, delete the image so YOLO doesn't treat it as negative sample incorrectly
        out_img_path.unlink(missing_ok=True)

yaml_content = f"""path: {YOLO_DIR.absolute()}
train: images/train
val: images/val

names:
  0: digit
  1: electronic_sequence
"""
(YOLO_DIR / "dataset.yaml").write_text(yaml_content)
print(f"Generated dataset with {generated} images at {YOLO_DIR}")

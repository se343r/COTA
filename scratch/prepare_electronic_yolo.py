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
        if "yolo" not in str(p) and "scratch" not in str(p):
            return p
    return None

YOLO_DIR = BASE_DIR / "yolo_electronic_dataset"
if YOLO_DIR.exists():
    shutil.rmtree(YOLO_DIR)

YOLO_DIR.mkdir()
(YOLO_DIR / "images" / "train").mkdir(parents=True)
(YOLO_DIR / "images" / "val").mkdir(parents=True)
(YOLO_DIR / "labels" / "train").mkdir(parents=True)
(YOLO_DIR / "labels" / "val").mkdir(parents=True)

json_files = list(DIGITS_DIR.glob("*.json"))
elec_files = []
for jf in json_files:
    with open(jf, "r") as f:
        data = json.load(f)
    if data.get("class_id", -1) == 1:
        elec_files.append(jf)

np.random.seed(42)
np.random.shuffle(elec_files)
split_idx = int(len(elec_files) * 0.8)

generated = 0

for i, jp in enumerate(elec_files):
    split = "train" if i < split_idx else "val"
    
    with open(jp, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    digits = data.get("digits", [])
    if not digits:
        continue # SKIP EARLY BEFORE SAVING IMAGE!
        
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
    out_lbl_path = YOLO_DIR / "labels" / split / f"{jp.stem}.txt"
    lines = []
    
    for d in digits:
        c = 0
        cx, cy, w, h = d["cx"], d["cy"], d["w"], d["h"]
        lines.append(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        
    if lines:
        cv2.imwrite(str(out_img_path), warped)
        out_lbl_path.write_text("\n".join(lines))
        generated += 1

yaml_content = f"""path: {YOLO_DIR.absolute()}
train: images/train
val: images/val

names:
  0: electronic_screen
"""
(YOLO_DIR / "dataset.yaml").write_text(yaml_content)
print(f"Generated dataset with {generated} images at {YOLO_DIR}")

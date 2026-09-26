import json, sys
from pathlib import Path

BASE_DIR = Path("/home/deist/Downloads/OCR/anh")
eval_file = BASE_DIR / "testocr_eval.json"
with open(eval_file) as f:
    eval_data = json.load(f)

for fname, d in eval_data.items():
    if d.get("type") == "trash": continue
    if d.get("actual_text") == "?" or not d.get("actual_text"): continue
    
    # find image
    img_path = None
    for p in (BASE_DIR / "filtered").rglob(fname):
        img_path = p
        break
    if not img_path:
        for p in BASE_DIR.rglob(fname):
            if "yolo" not in str(p) and "scratch" not in str(p):
                img_path = p
                break
    if not img_path:
        print(f"NOT FOUND: {fname}")

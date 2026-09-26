"""
Rebuild yolo_digits_dataset including half-digit (is_between=True).
Half-digit vẫn là class 0 — chỉ cần BBox detect được, không cần class riêng.
"""
import json, shutil
from pathlib import Path
import cv2, numpy as np

BASE_DIR   = Path("/home/deist/Downloads/OCR/anh")
DIGITS_DIR = BASE_DIR / "crops" / "digits_data"
YOLO_DIR   = BASE_DIR / "yolo_digits_dataset"

def find_image(name):
    for p in (BASE_DIR / "filtered").rglob(name):
        if p.is_file(): return p
    for p in BASE_DIR.rglob(name):
        if "yolo" not in str(p) and "scratch" not in str(p): return p
    return None

if YOLO_DIR.exists(): shutil.rmtree(YOLO_DIR)
for sub in ["images/train","images/val","labels/train","labels/val"]:
    (YOLO_DIR / sub).mkdir(parents=True)

json_files = list(DIGITS_DIR.glob("*.json"))
rng = np.random.default_rng(42)
rng.shuffle(json_files)
split_idx = int(len(json_files) * 0.8)

stats = {"total":0,"clean":0,"half":0,"skipped":0}

for i, jp in enumerate(json_files):
    split = "train" if i < split_idx else "val"
    with open(jp) as f: data = json.load(f)

    img_path = find_image(data["image"])
    if not img_path: stats["skipped"]+=1; continue
    img = cv2.imread(str(img_path))
    if img is None: stats["skipped"]+=1; continue

    pts = np.array(data["corners"], dtype=np.float32)
    pad = 20
    max_w = max(int(np.linalg.norm(pts[0]-pts[1])), int(np.linalg.norm(pts[2]-pts[3])))
    max_h = max(int(np.linalg.norm(pts[1]-pts[2])), int(np.linalg.norm(pts[0]-pts[3])))
    dst = np.array([[pad,pad],[pad+max_w,pad],[pad+max_w,pad+max_h],[pad,pad+max_h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (max_w+2*pad, max_h+2*pad))

    cls_id = data.get("class_id", 0)
    lines = []
    for d in data.get("digits", []):
        label = d.get("label", "")
        if label in ("unreadable", "", "?"): continue   # bỏ không đọc được
        # is_between => include (previously excluded)
        c = 0 if cls_id == 0 else 1
        lines.append(f"{c} {d['cx']:.6f} {d['cy']:.6f} {d['w']:.6f} {d['h']:.6f}")
        stats["total"] += 1
        if d.get("is_between"): stats["half"] += 1
        else: stats["clean"] += 1

    if lines:
        cv2.imwrite(str(YOLO_DIR/"images"/split/f"{jp.stem}.jpg"), warped)
        (YOLO_DIR/"labels"/split/f"{jp.stem}.txt").write_text("\n".join(lines))
    else:
        stats["skipped"] += 1

(YOLO_DIR/"dataset.yaml").write_text(f"""path: {YOLO_DIR.absolute()}
train: images/train
val: images/val

names:
  0: digit
  1: electronic_sequence
""")

print("="*50)
print(f"Dataset: {YOLO_DIR}")
print(f"  Total boxes  : {stats['total']}")
print(f"  Clean digits : {stats['clean']}")
print(f"  Half-digits  : {stats['half']}  <- previously excluded")
print(f"  Skipped imgs : {stats['skipped']}")
print(f"  Train imgs   : {len(list((YOLO_DIR/'images'/'train').glob('*.jpg')))}")
print(f"  Val imgs     : {len(list((YOLO_DIR/'images'/'val').glob('*.jpg')))}")
print("="*50)

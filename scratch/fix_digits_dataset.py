import json
import shutil
from pathlib import Path
import numpy as np

BASE_DIR = Path("/home/deist/Downloads/OCR/anh")
YOLO_DIGITS_DIR = BASE_DIR / "yolo_digits_dataset"

if YOLO_DIGITS_DIR.exists():
    shutil.rmtree(YOLO_DIGITS_DIR)

for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
    (YOLO_DIGITS_DIR / sub).mkdir(parents=True, exist_ok=True)

ocr_labels_dir = BASE_DIR / "ocr_dataset" / "labels"
ocr_crops_dir = BASE_DIR / "ocr_dataset" / "crops"
json_files = sorted(list(ocr_labels_dir.glob("*.json")))

rng = np.random.default_rng(42)
rng.shuffle(json_files)
split_idx = int(len(json_files) * 0.85)

total_boxes = 0
for i, jf in enumerate(json_files):
    split = "train" if i < split_idx else "val"
    with open(jf, encoding="utf-8") as f:
        item = json.load(f)

    crop_img_path = ocr_crops_dir / f"{jf.stem}.jpg"
    if not crop_img_path.exists():
        continue

    lines = []
    for d in item.get("digits", []):
        x, y, w, h = d["x"], d["y"], d["w"], d["h"]
        cx = x + w / 2.0
        cy = y + h / 2.0
        nw = w
        nh = h
        # Ensure within [0, 1]
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        nw = max(0.001, min(1.0, nw))
        nh = max(0.001, min(1.0, nh))
        lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        total_boxes += 1

    if lines:
        shutil.copy(crop_img_path, YOLO_DIGITS_DIR / "images" / split / f"{jf.stem}.jpg")
        (YOLO_DIGITS_DIR / "labels" / split / f"{jf.stem}.txt").write_text("\n".join(lines))

(YOLO_DIGITS_DIR / "dataset.yaml").write_text(f"""path: {YOLO_DIGITS_DIR.absolute()}
train: images/train
val: images/val

names:
  0: digit
""")

print("ĐÃ SỬA XONG yolo_digits_dataset:")
print(f"  Tổng boxes: {total_boxes}")
print(f"  Train: {len(list((YOLO_DIGITS_DIR/'images'/'train').glob('*.jpg')))} ảnh")
print(f"  Val  : {len(list((YOLO_DIGITS_DIR/'images'/'val').glob('*.jpg')))} ảnh")

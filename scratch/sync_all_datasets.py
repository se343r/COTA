import json
import shutil
from pathlib import Path
import cv2
import numpy as np

BASE_DIR = Path("/home/deist/Downloads/OCR/anh")
VALIDATE1_DIR = Path("/home/deist/Downloads/OCR/testocr")
VALIDATE1_EVAL = BASE_DIR / "validate1_eval.json"

import sys
sys.path.append(str(BASE_DIR))
from app import _load_model

print("="*60)
print("1. ĐỒNG BỘ DỮ LIỆU TẬP OBB (yolo_dataset)")
print("="*60)

with open(VALIDATE1_EVAL, encoding="utf-8") as f:
    v1_data = json.load(f)

obb_model = _load_model()
yolo_ds_dir = BASE_DIR / "yolo_dataset"
yolo_train_img = yolo_ds_dir / "images" / "train"
yolo_train_lbl = yolo_ds_dir / "labels" / "train"
yolo_train_img.mkdir(parents=True, exist_ok=True)
yolo_train_lbl.mkdir(parents=True, exist_ok=True)

for c in yolo_ds_dir.rglob("*.cache"):
    c.unlink(missing_ok=True)

obb_added = 0
for fname, d in v1_data.items():
    mtype = d.get("type")
    if mtype not in ("mechanical", "electronic"):
        continue
    cls_id = 0 if mtype == "mechanical" else 1

    img_path = VALIDATE1_DIR / fname
    if not img_path.exists():
        continue
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    H, W = img.shape[:2]

    quad = None
    if d.get("quad") and len(d["quad"]) == 4:
        quad = d["quad"]
    else:
        res = obb_model(img, conf=0.15, verbose=False)
        if len(res[0].obb) > 0:
            best_obb = max(res[0].obb, key=lambda x: x.conf[0].item())
            quad = best_obb.xyxyxyxy[0].tolist()

    if not quad or len(quad) != 4:
        continue

    coords = []
    for p in quad:
        coords.append(max(0.0, min(1.0, p[0] / W)))
        coords.append(max(0.0, min(1.0, p[1] / H)))

    line = f"{cls_id} " + " ".join(f"{v:.6f}" for v in coords)
    stem = Path(fname).stem
    shutil.copy(img_path, yolo_train_img / fname)
    (yolo_train_lbl / f"{stem}.txt").write_text(line)
    obb_added += 1

print(f"Đã thêm {obb_added} ảnh và nhãn OBB vào yolo_dataset/ (train split).")
print(f"Tổng ảnh OBB train: {len(list(yolo_train_img.glob('*.jpg')))}, val: {len(list((yolo_ds_dir/'images'/'val').glob('*.jpg')))}")


print("\n" + "="*60)
print("2. ĐỒNG BỘ DỮ LIỆU BBOX CÔNG TƠ CƠ (yolo_digits_dataset)")
print("="*60)

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
    img = cv2.imread(str(crop_img_path))
    if img is None:
        continue
    H, W = img.shape[:2]

    lines = []
    for d in item.get("digits", []):
        x, y, w, h = d["x"], d["y"], d["w"], d["h"]
        cx = (x + w / 2.0) / W
        cy = (y + h / 2.0) / H
        nw = w / W
        nh = h / H
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

print(f"Đã rebuild yolo_digits_dataset:")
print(f"  Tổng boxes: {total_boxes}")
print(f"  Train: {len(list((YOLO_DIGITS_DIR/'images'/'train').glob('*.jpg')))} ảnh")
print(f"  Val  : {len(list((YOLO_DIGITS_DIR/'images'/'val').glob('*.jpg')))} ảnh")


print("\n" + "="*60)
print("3. ĐỒNG BỘ DỮ LIỆU BBOX CÔNG TƠ ĐIỆN TỬ (yolo_electronic_dataset)")
print("="*60)

YOLO_ELEC_DIR = BASE_DIR / "yolo_electronic_dataset"
if YOLO_ELEC_DIR.exists():
    shutil.rmtree(YOLO_ELEC_DIR)

for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
    (YOLO_ELEC_DIR / sub).mkdir(parents=True, exist_ok=True)

digits_data_dir = BASE_DIR / "crops" / "digits_data"
old_elec_files = []
for jf in digits_data_dir.glob("*.json"):
    with open(jf) as f:
        dt = json.load(f)
    if dt.get("class_id") == 1 and dt.get("digits"):
        old_elec_files.append(jf)

rng = np.random.default_rng(42)
rng.shuffle(old_elec_files)
split_idx = int(len(old_elec_files) * 0.85)

def find_image(name):
    for p in (BASE_DIR / "filtered").rglob(name):
        if p.is_file(): return p
    for p in BASE_DIR.rglob(name):
        if "yolo" not in str(p) and "scratch" not in str(p) and "parseq" not in str(p):
            if p.is_file(): return p
    return None

elec_boxes_count = 0
for i, jf in enumerate(old_elec_files):
    split = "train" if i < split_idx else "val"
    with open(jf, encoding="utf-8") as f:
        dt = json.load(f)
    img_path = find_image(dt["image"])
    if not img_path: continue
    img = cv2.imread(str(img_path))
    if img is None: continue

    pts = np.array(dt["corners"], dtype=np.float32)
    pad = 20
    max_w = max(int(np.linalg.norm(pts[0]-pts[1])), int(np.linalg.norm(pts[2]-pts[3])))
    max_h = max(int(np.linalg.norm(pts[1]-pts[2])), int(np.linalg.norm(pts[0]-pts[3])))
    dst = np.array([[pad,pad],[pad+max_w,pad],[pad+max_w,pad+max_h],[pad,pad+max_h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (max_w + 2*pad, max_h + 2*pad))

    lines = []
    for d in dt.get("digits", []):
        lines.append(f"0 {d['cx']:.6f} {d['cy']:.6f} {d['w']:.6f} {d['h']:.6f}")
        elec_boxes_count += 1

    if lines:
        cv2.imwrite(str(YOLO_ELEC_DIR / "images" / split / f"{jf.stem}.jpg"), warped)
        (YOLO_ELEC_DIR / "labels" / split / f"{jf.stem}.txt").write_text("\n".join(lines))

val1_elec_added = 0
for fname, d in v1_data.items():
    if d.get("type") != "electronic":
        continue
    boxes = d.get("boxes")
    if not boxes:
        continue
    img_path = VALIDATE1_DIR / fname
    if not img_path.exists(): continue
    img = cv2.imread(str(img_path))
    if img is None: continue

    quad = d.get("quad")
    if not quad or len(quad) != 4:
        res = obb_model(img, conf=0.15, verbose=False)
        if len(res[0].obb) > 0:
            best_obb = max(res[0].obb, key=lambda x: x.conf[0].item())
            quad = best_obb.xyxyxyxy[0].tolist()

    if not quad or len(quad) != 4:
        continue

    pts_sum = [p[0] + p[1] for p in quad]
    pts_diff = [p[0] - p[1] for p in quad]
    tl = quad[np.argmin(pts_sum)]
    br = quad[np.argmax(pts_sum)]
    tr = quad[np.argmax(pts_diff)]
    bl = quad[np.argmin(pts_diff)]
    ordered_quad = [tl, tr, br, bl]

    pts = np.array(ordered_quad, dtype=np.float32)
    pad = 20
    max_w = max(20, int(np.linalg.norm(pts[0]-pts[1])), int(np.linalg.norm(pts[2]-pts[3])))
    max_h = max(20, int(np.linalg.norm(pts[1]-pts[2])), int(np.linalg.norm(pts[0]-pts[3])))
    dst = np.array([[pad, pad], [pad + max_w, pad], [pad + max_w, pad + max_h], [pad, pad + max_h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (max_w + 2*pad, max_h + 2*pad))
    W_w, H_w = warped.shape[1], warped.shape[0]

    lines = []
    for bx in boxes:
        x1, y1, x2, y2 = bx["x1"], bx["y1"], bx["x2"], bx["y2"]
        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0 or bh <= 0: continue
        cx = (x1 + bw / 2.0) / W_w
        cy = (y1 + bh / 2.0) / H_w
        nw = bw / W_w
        nh = bh / H_w
        lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        elec_boxes_count += 1

    if lines:
        stem = Path(fname).stem
        cv2.imwrite(str(YOLO_ELEC_DIR / "images" / "train" / f"{stem}.jpg"), warped)
        (YOLO_ELEC_DIR / "labels" / "train" / f"{stem}.txt").write_text("\n".join(lines))
        val1_elec_added += 1

(YOLO_ELEC_DIR / "dataset.yaml").write_text(f"""path: {YOLO_ELEC_DIR.absolute()}
train: images/train
val: images/val

names:
  0: electronic_screen
""")

print(f"Đã rebuild yolo_electronic_dataset (thêm {val1_elec_added} ảnh từ Validate 1):")
print(f"  Tổng boxes: {elec_boxes_count}")
print(f"  Train: {len(list((YOLO_ELEC_DIR/'images'/'train').glob('*.jpg')))} ảnh")
print(f"  Val  : {len(list((YOLO_ELEC_DIR/'images'/'val').glob('*.jpg')))} ảnh")


print("\n" + "="*60)
print("4. ĐỒNG BỘ DỮ LIỆU OCR CÔNG TƠ ĐIỆN TỬ (parseq_data)")
print("="*60)

parseq_train_img_dir = BASE_DIR / "parseq_data" / "train" / "images"
parseq_train_gt_file = BASE_DIR / "parseq_data" / "train" / "gt.txt"
parseq_train_img_dir.mkdir(parents=True, exist_ok=True)

existing_gt = {}
if parseq_train_gt_file.exists():
    with open(parseq_train_gt_file, "r", encoding="utf-8") as gf:
        for line in gf:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                existing_gt[parts[0]] = parts[1]

parseq_added = 0
for fname, d in v1_data.items():
    if d.get("type") != "electronic":
        continue
    details = d.get("digit_details", [])
    if not details or not details[0].get("actual"):
        continue
    raw_reading = details[0]["actual"]
    clean_reading = "".join(c for c in raw_reading if c.isdigit())
    if not clean_reading:
        continue

    boxes = d.get("boxes", [])
    if not boxes:
        continue

    img_path = VALIDATE1_DIR / fname
    if not img_path.exists(): continue
    img = cv2.imread(str(img_path))
    if img is None: continue

    quad = d.get("quad")
    if not quad or len(quad) != 4:
        res = obb_model(img, conf=0.15, verbose=False)
        if len(res[0].obb) > 0:
            best_obb = max(res[0].obb, key=lambda x: x.conf[0].item())
            quad = best_obb.xyxyxyxy[0].tolist()
    if not quad or len(quad) != 4:
        continue

    pts_sum = [p[0] + p[1] for p in quad]
    pts_diff = [p[0] - p[1] for p in quad]
    tl = quad[np.argmin(pts_sum)]
    br = quad[np.argmax(pts_sum)]
    tr = quad[np.argmax(pts_diff)]
    bl = quad[np.argmin(pts_diff)]
    ordered_quad = [tl, tr, br, bl]

    pts = np.array(ordered_quad, dtype=np.float32)
    pad = 20
    max_w = max(20, int(np.linalg.norm(pts[0]-pts[1])), int(np.linalg.norm(pts[2]-pts[3])))
    max_h = max(20, int(np.linalg.norm(pts[1]-pts[2])), int(np.linalg.norm(pts[0]-pts[3])))
    dst = np.array([[pad, pad], [pad + max_w, pad], [pad + max_w, pad + max_h], [pad, pad + max_h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (max_w + 2*pad, max_h + 2*pad))

    bx = boxes[0]
    x1, y1, x2, y2 = int(bx["x1"]), int(bx["y1"]), int(bx["x2"]), int(bx["y2"])
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(warped.shape[1], x2); y2 = min(warped.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        continue

    lcd_crop = warped[y1:y2, x1:x2]
    crop_fname = f"val1_{fname}"
    cv2.imwrite(str(parseq_train_img_dir / crop_fname), lcd_crop)
    existing_gt[crop_fname] = clean_reading
    parseq_added += 1

with open(parseq_train_gt_file, "w", encoding="utf-8") as gf:
    for k, v in existing_gt.items():
        gf.write(f"{k}\t{v}\n")

print(f"Đã thêm {parseq_added} mẫu đọc màn hình điện tử vào parseq_data/train/ (tổng mẫu train: {len(existing_gt)})")
print("="*60)
print("HOÀN THÀNH ĐỒNG BỘ 4 TẬP DỮ LIỆU!")

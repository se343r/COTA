import json, cv2, numpy as np
from pathlib import Path

BASE_DIR = Path("/home/deist/Downloads/OCR/anh")
VALIDATE1_DIR = Path("/home/deist/Downloads/OCR/testocr")
CROPS_TARGET = BASE_DIR / "ocr_dataset" / "crops"
LABELS_TARGET = BASE_DIR / "ocr_dataset" / "labels"
CROPS_TARGET.mkdir(parents=True, exist_ok=True)
LABELS_TARGET.mkdir(parents=True, exist_ok=True)

import sys
sys.path.append(str(BASE_DIR))
from app import _load_model, get_yolo_digits

with open(BASE_DIR / "validate1_eval.json", encoding="utf-8") as f:
    eval_data = json.load(f)

obb_model = _load_model()
model_bbox = get_yolo_digits()

exported_images = 0
clean_digits_count = 0
decimal_digits_count = 0
half_digits_count = 0

for fname, d in eval_data.items():
    if d.get("type") != "mechanical" or not d.get("digits_bbox_correct"):
        continue
        
    img_path = VALIDATE1_DIR / fname
    if not img_path.exists():
        continue
        
    img = cv2.imread(str(img_path))
    if img is None:
        continue
        
    # Determine quad (either custom quad or YOLO OBB)
    quad = None
    if d.get("quad") and len(d["quad"]) == 4:
        quad = [tuple(p) for p in d["quad"]]
    else:
        res_obb = obb_model(img, conf=0.15, verbose=False)
        if len(res_obb[0].obb) > 0:
            best_obb = max(res_obb[0].obb, key=lambda x: x.conf[0].item())
            quad = [tuple(p) for p in best_obb.xyxyxyxy[0].tolist()]
            
    if not quad or len(quad) != 4:
        print(f"Skipping {fname}: No valid quad found.")
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
    w1 = int(np.linalg.norm(pts[0] - pts[1]))
    w2 = int(np.linalg.norm(pts[2] - pts[3]))
    h1 = int(np.linalg.norm(pts[1] - pts[2]))
    h2 = int(np.linalg.norm(pts[0] - pts[3]))
    max_w = max(20, max(w1, w2))
    max_h = max(20, max(h1, h2))
    
    dst = np.array([
        [pad, pad],
        [pad + max_w, pad],
        [pad + max_w, pad + max_h],
        [pad, pad + max_h]
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (max_w + 2*pad, max_h + 2*pad))
    W_warp, H_warp = warped.shape[1], warped.shape[0]
    
    # Get boxes (custom or YOLO)
    boxes = []
    if d.get("boxes") and len(d["boxes"]) > 0:
        boxes = d["boxes"]
    else:
        res_bbox = model_bbox(warped, conf=0.25, iou=0.45, verbose=False)
        for bx in res_bbox[0].boxes:
            boxes.append({
                "x1": float(bx.xyxy[0][0]), "y1": float(bx.xyxy[0][1]),
                "x2": float(bx.xyxy[0][2]), "y2": float(bx.xyxy[0][3]),
            })
    boxes.sort(key=lambda b: b["x1"])
    
    details = d.get("digit_details", [])
    stem = Path(fname).stem
    
    digit_entries = []
    for idx, bx in enumerate(boxes):
        if idx < len(details):
            det = details[idx]
            if det.get("is_spurious"):
                continue # Skip spurious boxes!
            lbl = str(det.get("actual", "")).strip()
            if not lbl or not lbl.isdigit():
                continue
            is_half = bool(det.get("is_half", False))
            is_dec = bool(det.get("is_decimal", False))
            
            x_norm = max(0.0, min(1.0, bx["x1"] / W_warp))
            y_norm = max(0.0, min(1.0, bx["y1"] / H_warp))
            w_norm = max(0.001, min(1.0, (bx["x2"] - bx["x1"]) / W_warp))
            h_norm = max(0.001, min(1.0, (bx["y2"] - bx["y1"]) / H_warp))
            
            digit_entries.append({
                "digit_index": len(digit_entries),
                "x": x_norm,
                "y": y_norm,
                "w": w_norm,
                "h": h_norm,
                "label": lbl,
                "is_decimal": is_dec,
                "mid_transition": is_half
            })
            if is_half:
                half_digits_count += 1
            else:
                clean_digits_count += 1
                if is_dec:
                    decimal_digits_count += 1
                    
    if digit_entries:
        cv2.imwrite(str(CROPS_TARGET / f"{stem}.jpg"), warped)
        lbl_data = {
            "crop_id": stem,
            "source_image": fname,
            "digits": digit_entries
        }
        with open(LABELS_TARGET / f"{stem}.json", "w", encoding="utf-8") as out_f:
            json.dump(lbl_data, out_f, ensure_ascii=False, indent=2)
        exported_images += 1

print(f"=== KẾT QUẢ XUẤT VALIDATE 1 VÀO OCR_DATASET ===")
print(f"Số ảnh xuất thành công: {exported_images}")
print(f"Tổng chữ số sạch (0-9): {clean_digits_count}")
print(f"  Trong đó số đỏ (is_decimal=True): {decimal_digits_count}")
print(f"Số chữ số lưng chừng (mid_transition=True): {half_digits_count}")

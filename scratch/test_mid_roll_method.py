import sys
from pathlib import Path
BASE_DIR = Path("/home/deist/Downloads/OCR/anh")
sys.path.insert(0, str(BASE_DIR))

import json
import glob
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from collections import Counter

from mid_roll_detector import detect_mid_roll, is_adjacent, RISKY_PAIRS
from app import TinyDigitCNN, LetterboxPad, get_ocr_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("="*70)
print("1. KHỞI TẠO MÔ HÌNH VÀ BỘ TEST")
print("="*70)

ocr_model, ocr_transform = get_ocr_model()
if ocr_model is None:
    raise RuntimeError("Không thể nạp TinyDigitCNN!")

# Thu thập tất cả các mẫu chữ số trong ocr_dataset:
# Phân loại thành: mid_roll_samples (150 mẫu) và clean_samples (2585 mẫu)
crops_dir = BASE_DIR / "ocr_dataset" / "crops"
labels_dir = BASE_DIR / "ocr_dataset" / "labels"

mid_roll_samples = []
clean_samples = []

for jf in labels_dir.glob("*.json"):
    with open(jf, encoding="utf-8") as f:
        data = json.load(f)
    crop_id = data["crop_id"]
    crop_path = crops_dir / f"{crop_id}.jpg"
    if not crop_path.exists():
        continue

    for d in data.get("digits", []):
        lbl = d.get("label", "")
        if "x" not in d or "w" not in d:
            continue
        is_mid = bool(d.get("mid_transition")) or bool(d.get("is_half"))
        item = {
            "crop_id": crop_id,
            "crop_path": str(crop_path),
            "digit": d,
            "label": lbl,
            "is_decimal": d.get("is_decimal", False),
            "is_mid": is_mid
        }
        if is_mid:
            mid_roll_samples.append(item)
        else:
            clean_samples.append(item)

print(f"Tổng số mẫu từ ocr_dataset:")
print(f"  - Mid-roll (Half-digit) thực tế: {len(mid_roll_samples)} mẫu")
print(f"  - Clean digit thực tế: {len(clean_samples)} mẫu")


def predict_digit_probs(crop_img_path, d):
    """Cắt vùng chữ số và chạy TinyDigitCNN để lấy dict softmax probabilities."""
    try:
        img = Image.open(crop_img_path).convert("RGB")
        W, H = img.size
        x1 = int(round(d["x"] * W))
        y1 = int(round(d["y"] * H))
        x2 = int(round((d["x"] + d["w"]) * W))
        y2 = int(round((d["y"] + d["h"]) * H))
        x1 = max(0, x1); y1 = max(0, y1); x2 = min(W, x2); y2 = min(H, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = img.crop((x1, y1, x2, y2))
        inp = ocr_transform(crop).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = ocr_model(inp)
            probs = torch.nn.functional.softmax(logits, dim=1)[0].cpu().tolist()
        return {str(i): probs[i] for i in range(10)}
    except Exception as e:
        return None


print("\n" + "="*70)
print("2. ĐÁNH GIÁ TRÊN TẬP MID-ROLL THỰC TẾ (150 mẫu)")
print("="*70)

mid_detected_count = 0
adjacent_count = 0
margins = []
pair_counter = Counter()
results_mid = []

for s in mid_roll_samples:
    probs = predict_digit_probs(s["crop_path"], s["digit"])
    if probs is None:
        continue
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    top1, p1 = ranked[0]
    top2, p2 = ranked[1]
    margin = p1 - p2
    margins.append(margin)
    adj = is_adjacent(top1, top2)
    if adj:
        adjacent_count += 1
        pair_key = tuple(sorted([top1, top2]))
        pair_counter[pair_key] += 1

    r = detect_mid_roll(probs, margin_threshold=0.35, risky_margin_threshold=0.15)
    results_mid.append((s, probs, r, margin, adj))
    if r.is_mid_roll:
        mid_detected_count += 1

print(f"Tổng số mẫu mid-roll kiểm tra thành công: {len(results_mid)}")
print(f"• Số mẫu có Top-1 & Top-2 LIỀN KỀ vòng tròn: {adjacent_count}/{len(results_mid)} ({adjacent_count/len(results_mid)*100:.1f}%)")
print(f"• Số mẫu được detect_mid_roll gắn nhãn mid-roll (Recall): {mid_detected_count}/{len(results_mid)} ({mid_detected_count/len(results_mid)*100:.1f}%)")
print(f"• Phân bố margin trung bình của các ca mid-roll: {sum(margins)/len(margins):.3f} (Min: {min(margins):.3f}, Max: {max(margins):.3f})")

print("\nTop các cặp số liền kề xuất hiện nhiều nhất khi mid-roll:")
for pair, cnt in pair_counter.most_common(10):
    print(f"  Cặp {pair}: {cnt} lần")


print("\n" + "="*70)
print("3. ĐÁNH GIÁ FALSE POSITIVE TRÊN TẬP CLEAN DIGITS (2.585 mẫu)")
print("="*70)

clean_fp_count = 0
clean_adjacent_count = 0
clean_margins = []
results_clean = []

for s in clean_samples:
    probs = predict_digit_probs(s["crop_path"], s["digit"])
    if probs is None:
        continue
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    top1, p1 = ranked[0]
    top2, p2 = ranked[1]
    margin = p1 - p2
    clean_margins.append(margin)
    adj = is_adjacent(top1, top2)
    if adj:
        clean_adjacent_count += 1

    r = detect_mid_roll(probs, margin_threshold=0.35, risky_margin_threshold=0.15)
    results_clean.append((s, probs, r, margin, adj))
    if r.is_mid_roll:
        clean_fp_count += 1

print(f"Tổng số mẫu clean kiểm tra thành công: {len(results_clean)}")
print(f"• Số mẫu clean có Top-1 & Top-2 liền kề: {clean_adjacent_count}/{len(results_clean)} ({clean_adjacent_count/len(results_clean)*100:.1f}%)")
print(f"• Số mẫu clean bị gán nhầm là mid-roll (False Positive Rate): {clean_fp_count}/{len(results_clean)} ({clean_fp_count/len(results_clean)*100:.2f}%)")
print(f"• Độ đặc hiệu (Specificity): {(len(results_clean)-clean_fp_count)/len(results_clean)*100:.2f}%")
print(f"• Margin trung bình của các ca clean: {sum(clean_margins)/len(clean_margins):.3f}")


print("\n" + "="*70)
print("4. KIỂM THỬ TRÊN TẬP VALIDATE 1 VÀ VALIDATE 2 THỰC TẾ")
print("="*70)

def test_eval_json(eval_path, img_dir, name):
    print(f"\n--- Kiểm tra {name} ({eval_path.name}) ---")
    with open(eval_path, encoding="utf-8") as f:
        data = json.load(f)
    
    half_cases = 0
    detected_half = 0
    resolved_matches = 0
    
    from app import _load_model, get_yolo_digits
    obb_model = _load_model()
    model_bbox = get_yolo_digits()
    
    for fname, item in data.items():
        if item.get("type") != "mechanical":
            continue
        details = item.get("digit_details", [])
        if not any(d.get("is_half") for d in details):
            continue
            
        img_p = img_dir / fname
        if not img_p.exists(): continue
        import cv2, numpy as np
        img = cv2.imread(str(img_p))
        if img is None: continue
        
        # Warp with quad
        quad = item.get("quad")
        if not quad:
            res_o = obb_model(img, conf=0.15, verbose=False)
            if len(res_o[0].obb) > 0:
                best_o = max(res_o[0].obb, key=lambda x: x.conf[0].item())
                quad = [tuple(p) for p in best_o.xyxyxyxy[0].tolist()]
        if not quad: continue
        
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
        
        boxes = item.get("boxes", [])
        if not boxes:
            res_b = model_bbox(warped, conf=0.25, iou=0.45, verbose=False)
            for bx in res_b[0].boxes:
                boxes.append({"x1": float(bx.xyxy[0][0]), "y1": float(bx.xyxy[0][1]), "x2": float(bx.xyxy[0][2]), "y2": float(bx.xyxy[0][3])})
        boxes.sort(key=lambda b: b["x1"])
        
        for idx, det in enumerate(details):
            if det.get("is_half") and idx < len(boxes):
                half_cases += 1
                bx = boxes[idx]
                x1, y1, x2, y2 = int(bx["x1"]), int(bx["y1"]), int(bx["x2"]), int(bx["y2"])
                x1 = max(0, x1); y1 = max(0, y1); x2 = min(warped.shape[1], x2); y2 = min(warped.shape[0], y2)
                if x2 <= x1 or y2 <= y1: continue
                c_crop = warped[y1:y2, x1:x2]
                c_pil = Image.fromarray(cv2.cvtColor(c_crop, cv2.COLOR_BGR2RGB))
                inp = ocr_transform(c_pil).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    logits = ocr_model(inp)
                    p_list = torch.nn.functional.softmax(logits, dim=1)[0].cpu().tolist()
                p_dict = {str(i): p_list[i] for i in range(10)}
                res_mid = detect_mid_roll(p_dict)
                act_str = det.get("actual", "")
                
                print(f"  [{fname[:15]}... #{idx+1}] Nhãn tay='{act_str}' | Top-1={res_mid.top1_label} ({res_mid.top1_prob:.2f}), Top-2={res_mid.top2_label} ({res_mid.top2_prob:.2f}), Margin={res_mid.margin:.2f}")
                print(f"      -> Mid-roll: {res_mid.is_mid_roll} | Resolved: {res_mid.resolved_value} | {res_mid.reason}")
                
                if res_mid.is_mid_roll:
                    detected_half += 1
                if res_mid.resolved_value == act_str:
                    resolved_matches += 1

    print(f"\nKết quả trên {name}:")
    print(f"  • Tổng số ca half-digit: {half_cases}")
    print(f"  • Số ca detect được mid-roll: {detected_half}/{half_cases} ({detected_half/max(1,half_cases)*100:.1f}%)")
    print(f"  • Số ca resolved_value trùng khớp nhãn tay: {resolved_matches}/{half_cases} ({resolved_matches/max(1,half_cases)*100:.1f}%)")

test_eval_json(BASE_DIR / "validate1_eval.json", Path("/home/deist/Downloads/OCR/testocr"), "Validate 1")
test_eval_json(BASE_DIR / "validate2_eval.json", Path("/home/deist/Downloads/OCR/Validate 2"), "Validate 2")

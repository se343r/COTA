import json, cv2, torch, numpy as np, sys, shutil
from pathlib import Path

BASE_DIR = Path("/home/deist/Downloads/OCR/anh")
ARTIFACT_DIR = Path("/home/deist/.gemini/antigravity/brain/c327d66a-ae54-4b63-a997-fec47b910363")
TEST_IMAGES_DIR = Path("/home/deist/Downloads/OCR/testocr")

sys.path.append(str(BASE_DIR))
from app import get_yolo_digits, _load_model, get_ocr_model

with open(BASE_DIR / "scratch" / "full_stats.json") as f:
    stats = json.load(f)

errors = stats.get("high_conf_errors", [])

html = """# Phân tích các ca đoán sai với Confidence cao (>60%)

Dưới đây là các chữ số mà mô hình BBox đã cắt đúng, nhưng mô hình phân loại (TinyDigit) đoán sai với **độ tự tin rất cao (trên 60%)**. Việc kiểm tra những ca này giúp bạn nhận diện các điểm mù của mô hình hiện tại.

"""

if not errors:
    html += "*Tuyệt vời! Không có ca nào đoán sai với confidence > 60%.*"
else:
    # Group by image
    grouped = {}
    for err in errors:
        fname = err["img"]
        if fname not in grouped:
            grouped[fname] = []
        grouped[fname].append(err)
        
    for fname, errs in grouped.items():
        img_path = TEST_IMAGES_DIR / fname
        if not img_path.exists(): continue
        
        orig_dest = ARTIFACT_DIR / fname
        shutil.copy(img_path, orig_dest)
        
        # Inference just to get the BBox crops
        img = cv2.imread(str(img_path))
        obb_model = _load_model()
        res_obb = obb_model(img, conf=0.25, verbose=False)
        best_obb = max(res_obb[0].obb, key=lambda x: x.conf[0].item())
        pts_raw = best_obb.xyxyxyxy[0].cpu().numpy()
        rect = np.zeros((4, 2), dtype="float32")
        s = pts_raw.sum(axis=1)
        rect[0] = pts_raw[np.argmin(s)]
        rect[2] = pts_raw[np.argmax(s)]
        diff = np.diff(pts_raw, axis=1)
        rect[1] = pts_raw[np.argmin(diff)]
        rect[3] = pts_raw[np.argmax(diff)]
        pts = rect
        pad = 20
        max_w = max(int(np.linalg.norm(pts[0]-pts[1])), int(np.linalg.norm(pts[2]-pts[3])))
        max_h = max(int(np.linalg.norm(pts[1]-pts[2])), int(np.linalg.norm(pts[0]-pts[3])))
        dst = np.array([[pad,pad],[pad+max_w,pad],[pad+max_w,pad+max_h],[pad,pad+max_h]], np.float32)
        M = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(img, M, (max_w+2*pad, max_h+2*pad))
        
        model_bbox = get_yolo_digits()
        res_bbox = model_bbox(warped, conf=0.35, iou=0.45, verbose=False)
        boxes = []
        for bx in res_bbox[0].boxes:
            boxes.append({
                "x1": float(bx.xyxy[0][0]), "y1": float(bx.xyxy[0][1]),
                "x2": float(bx.xyxy[0][2]), "y2": float(bx.xyxy[0][3]),
            })
        boxes.sort(key=lambda b: b["x1"])
        
        margin = pad
        filtered = []
        for bx in boxes:
            cx = (bx["x1"] + bx["x2"]) / 2
            cy = (bx["y1"] + bx["y2"]) / 2
            if cx >= pad - margin and cx <= pad + max_w + margin and cy >= pad - margin and cy <= pad + max_h + margin:
                filtered.append(bx)
        boxes = filtered
        
        html += f"## Ảnh: `{fname}`\n\n"
        html += f"![Ảnh gốc]({orig_dest.absolute()})\n\n"
        
        html += "| Vị trí | Ảnh Crop | Nhãn đúng | Dự đoán sai | Độ tự tin (Conf) |\n"
        html += "|---|---|---|---|---|\n"
        
        for err in errs:
            idx = err["digit_index"]
            if idx < len(boxes):
                bx = boxes[idx]
                x1, y1, x2, y2 = int(bx["x1"]), int(bx["y1"]), int(bx["x2"]), int(bx["y2"])
                x1 = max(0, x1); y1 = max(0, y1); x2 = min(warped.shape[1], x2); y2 = min(warped.shape[0], y2)
                crop = warped[y1:y2, x1:x2]
                crop_fname = f"hc_err_{idx}_{fname}"
                cv2.imwrite(str(ARTIFACT_DIR / crop_fname), crop)
                
                conf_pct = err['conf'] * 100
                html += f"| Chữ số thứ {idx+1} | ![{crop_fname}]({(ARTIFACT_DIR / crop_fname).absolute()}) | **{err['actual_char']}** | **{err['pred_char']}** | {conf_pct:.2f}% |\n"
        
        html += "\n---\n\n"

with open(ARTIFACT_DIR / "high_conf_errors.md", "w") as f:
    f.write(html)

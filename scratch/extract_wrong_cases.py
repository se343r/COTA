import json, cv2, torch, numpy as np, sys, shutil
from pathlib import Path

BASE_DIR = Path("/home/deist/Downloads/OCR/anh")
ARTIFACT_DIR = Path("/home/deist/.gemini/antigravity/brain/c327d66a-ae54-4b63-a997-fec47b910363")
TEST_IMAGES_DIR = Path("/home/deist/Downloads/OCR/testocr")

sys.path.append(str(BASE_DIR))
from app import get_yolo_digits, _load_model, get_ocr_model

with open(BASE_DIR / "testocr_eval.json") as f:
    eval_data = json.load(f)

targets = {
    "134820": "024922",
    "090607": "060606",
    "306185": "300196"
}

html = """# Phân tích các ca OCR sai

Dưới đây là chi tiết các ca nhận diện sai nghiêm trọng. Tôi đã cắt sẵn từng chữ số (digit crop) mà mô hình BBox nhận diện được để bạn đối chiếu xem lỗi do BBox cắt sai/thiếu hay do mô hình phân loại (OCR) đoán sai.

"""

for fname, d in eval_data.items():
    actual = d.get("actual_text", "").replace(" ", "").replace("_", "")
    if actual in targets:
        img_path = TEST_IMAGES_DIR / fname
        if not img_path.exists(): continue
        
        orig_dest = ARTIFACT_DIR / fname
        shutil.copy(img_path, orig_dest)
        
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
        
        warped_fname = "warped_" + fname
        cv2.imwrite(str(ARTIFACT_DIR / warped_fname), warped)
        
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
        
        o_model, o_tf = get_ocr_model()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        html += f"## Ảnh: `{fname}`\n"
        html += f"- **Thực tế:** `{actual}`\n"
        html += f"- **Mô hình đoán:** `{targets[actual]}`\n\n"
        html += f"![Ảnh gốc]({orig_dest.absolute()})\n"
        html += f"![Ảnh cắt OBB]({(ARTIFACT_DIR / warped_fname).absolute()})\n\n"
        
        html += "| Digit Index | Kích thước (WxH) | Crop | Dự đoán |\n"
        html += "|---|---|---|---|\n"
        
        from PIL import Image
        for idx, bx in enumerate(boxes):
            x1, y1, x2, y2 = int(bx["x1"]), int(bx["y1"]), int(bx["x2"]), int(bx["y2"])
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(warped.shape[1], x2); y2 = min(warped.shape[0], y2)
            if x2 > x1 and y2 > y1:
                crop = warped[y1:y2, x1:x2]
                h, w = crop.shape[:2]
                crop_fname = f"crop_{idx}_{fname}"
                cv2.imwrite(str(ARTIFACT_DIR / crop_fname), crop)
                
                inp = o_tf(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
                with torch.no_grad():
                    probs = torch.nn.functional.softmax(o_model(inp), dim=1)
                    pred_char = str(probs.argmax(dim=1).item())
                    
                html += f"| {idx+1} | `{w} x {h}` | ![{crop_fname}]({(ARTIFACT_DIR / crop_fname).absolute()}) | **{pred_char}** |\n"
        
        html += "\n---\n\n"

with open(ARTIFACT_DIR / "wrong_predictions.md", "w") as f:
    f.write(html)

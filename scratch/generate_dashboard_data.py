import json
import os
from pathlib import Path

BASE_DIR = Path("/home/deist/Downloads/OCR/anh")
eval_file = BASE_DIR / "testocr_eval.json"
with open(eval_file) as f:
    eval_data = json.load(f)

# Initialize stats
stats = {
    "total": 0,
    "electronic": {"total": 0, "obb_ok": 0, "bbox_ok": 0, "ocr_ok": 0, "ocr_errors": []},
    "mechanical": {"total": 0, "obb_ok": 0, "bbox_ok": 0, "ocr_ok": 0, "ocr_errors": []},
    "trash": {"total": 0}
}

# Re-evaluate OCR correctness locally using the most up to date model
# Wait, no! The user's manual evaluation doesn't have the OCR predicted strings.
# I need to run inference to get the predicted strings.
import sys
sys.path.append(str(BASE_DIR))
from app import get_yolo_digits, get_yolo_electronic, get_ocr_model, get_parseq_model, _load_model, TEST_IMAGES_DIR
import cv2, torch, numpy as np
from PIL import Image

def infer_image(img_path, expected_cls):
    img = cv2.imread(str(img_path))
    if img is None: return None
    
    obb_model = _load_model()
    res_obb = obb_model(img, conf=0.25, verbose=False)
    if len(res_obb[0].obb) == 0: return None
    
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
    
    cls_id = 1 if expected_cls == "electronic" else 0
    
    model_bbox = get_yolo_electronic() if cls_id == 1 else get_yolo_digits()
    conf_thresh = 0.25 if cls_id == 1 else 0.35
    res_bbox = model_bbox(warped, conf=conf_thresh, iou=0.45, verbose=False)
    
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
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pred_str = ""
    
    if cls_id == 1:
        p_model, p_tf = get_parseq_model()
        for bx in boxes:
            x1, y1, x2, y2 = int(bx["x1"]), int(bx["y1"]), int(bx["x2"]), int(bx["y2"])
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(warped.shape[1], x2); y2 = min(warped.shape[0], y2)
            if x2 > x1 and y2 > y1 and p_model:
                crop = warped[y1:y2, x1:x2]
                inp = p_tf(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
                with torch.no_grad():
                    label, _ = p_model.tokenizer.decode(p_model(inp).softmax(-1))
                    pred_str += label[0]
        return pred_str
    else:
        o_model, o_tf = get_ocr_model()
        for bx in boxes:
            x1, y1, x2, y2 = int(bx["x1"]), int(bx["y1"]), int(bx["x2"]), int(bx["y2"])
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(warped.shape[1], x2); y2 = min(warped.shape[0], y2)
            if x2 > x1 and y2 > y1 and o_model:
                crop = warped[y1:y2, x1:x2]
                inp = o_tf(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
                with torch.no_grad():
                    probs = torch.nn.functional.softmax(o_model(inp), dim=1)
                    pred_str += str(probs.argmax(dim=1).item())
        return pred_str

import base64
def get_b64(path):
    with open(path, "rb") as image_file:
        return "data:image/jpeg;base64," + base64.b64encode(image_file.read()).decode('utf-8')

for fname, d in eval_data.items():
    stats["total"] += 1
    t = d.get("type")
    if t not in stats: stats[t] = {"total": 0, "obb_ok": 0, "bbox_ok": 0, "ocr_ok": 0, "ocr_errors": []}
    if t == "trash":
        stats["trash"]["total"] += 1
        continue
        
    stats[t]["total"] += 1
    if d.get("obb"): stats[t]["obb_ok"] += 1
    if d.get("digits_bbox_correct"): stats[t]["bbox_ok"] += 1
    
    img_path = TEST_IMAGES_DIR / fname
    if not img_path.exists(): continue
    
    actual = d.get("actual_text", "").replace(" ", "").replace("_", "")
    if actual == "?" or not actual: continue
    
    pred = infer_image(img_path, t)
    if pred is None: pred = ""
    
    if pred == actual:
        stats[t]["ocr_ok"] += 1
    else:
        # Save error info and a tiny thumbnail base64 for display
        stats[t]["ocr_errors"].append({
            "filename": fname,
            "actual": actual,
            "pred": pred,
            "img_b64": get_b64(img_path)
        })

with open("/home/deist/Downloads/OCR/anh/scratch/dashboard_data.json", "w") as f:
    json.dump(stats, f)


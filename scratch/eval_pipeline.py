import json, cv2, torch, numpy as np, sys
from pathlib import Path
from PIL import Image

sys.path.append("/home/deist/Downloads/OCR/anh")
from app import get_yolo_digits, get_yolo_electronic, get_ocr_model, get_parseq_model, _load_model, BASE_DIR, TEST_IMAGES_DIR

eval_file = BASE_DIR / "testocr_eval.json"
with open(eval_file) as f:
    eval_data = json.load(f)

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
    
    pad = 5
    max_w = max(int(np.linalg.norm(pts[0]-pts[1])), int(np.linalg.norm(pts[2]-pts[3])))
    max_h = max(int(np.linalg.norm(pts[1]-pts[2])), int(np.linalg.norm(pts[0]-pts[3])))
    dst = np.array([[pad,pad],[pad+max_w,pad],[pad+max_w,pad+max_h],[pad,pad+max_h]], np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (max_w+2*pad, max_h+2*pad))
    
    cls_id = 1 if expected_cls == "electronic" else 0
    
    model_bbox = get_yolo_electronic() if cls_id == 1 else get_yolo_digits()
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
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if cls_id == 1:
        p_model, p_tf = get_parseq_model()
        pred_str = ""
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
        pred_str = ""
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

stats = {"mechanical": {"total": 0, "correct": 0}, "electronic": {"total": 0, "correct": 0}, "other": {"total": 0, "correct": 0}}
results_list = []

for fname, d in eval_data.items():
    if d.get("type") == "trash": continue
    if d.get("actual_text") == "?" or not d.get("actual_text"): continue
    
    img_path = TEST_IMAGES_DIR / fname
    if not img_path.exists(): continue
    
    actual = d["actual_text"].replace(" ", "").replace("_", "")
    pred = infer_image(img_path, d["type"])
    if pred is None: pred = ""
    
    t = d["type"]
    stats[t]["total"] += 1
    if pred == actual:
        stats[t]["correct"] += 1
    else:
        print(f"[{t}] {fname}: Expected '{actual}', Got '{pred}'")
    
    results_list.append({
        "img": fname[:3] + "..." + fname[-6:],
        "actual": actual,
        "pred": pred,
        "type": t,
        "ok": pred == actual
    })

print("\n--- RESULTS ---")
for t in stats:
    tot = stats[t]["total"]
    if tot > 0:
        corr = stats[t]["correct"]
        print(f"{t.capitalize()}: {corr}/{tot} ({corr/tot*100:.1f}%)")

with open("/home/deist/Downloads/OCR/anh/scratch/eval_out.json", "w") as f:
    json.dump({"stats": stats, "details": results_list}, f)


import json, cv2, torch, numpy as np, sys
from pathlib import Path
from PIL import Image

sys.path.append("/home/deist/Downloads/OCR/anh")
from app import get_yolo_digits, get_yolo_electronic, get_ocr_model, get_parseq_model, _load_model, BASE_DIR

TEST_IMAGES_DIR = Path("/home/deist/Downloads/OCR/testocr")

with open("/home/deist/Downloads/OCR/anh/testocr_eval.json") as f:
    eval_data = json.load(f)

def infer_image(img_path, expected_cls):
    img = cv2.imread(str(img_path))
    if img is None: return None, []
    obb_model = _load_model()
    res_obb = obb_model(img, conf=0.25, verbose=False)
    if len(res_obb[0].obb) == 0: return None, []
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
    conf_scores = []
    if cls_id == 1:
        p_model, p_tf = get_parseq_model()
        for bx in boxes:
            x1, y1, x2, y2 = int(bx["x1"]), int(bx["y1"]), int(bx["x2"]), int(bx["y2"])
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(warped.shape[1], x2); y2 = min(warped.shape[0], y2)
            if x2 > x1 and y2 > y1 and p_model:
                crop = warped[y1:y2, x1:x2]
                inp = p_tf(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
                with torch.no_grad():
                    # PARSeq output doesn't easily give per-char conf in our simple snippet, we will fake it to 1.0 for now for electronic
                    label, _ = p_model.tokenizer.decode(p_model(inp).softmax(-1))
                    pred_str += label[0]
                    for _ in label[0]:
                        conf_scores.append(1.0)
        return pred_str, conf_scores
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
                    max_prob = probs.max(dim=1)[0].item()
                    pred_char = str(probs.argmax(dim=1).item())
                    pred_str += pred_char
                    conf_scores.append(max_prob)
        return pred_str, conf_scores

stats = {
    "total_images": len(eval_data),
    "obb_ok": 0,
    "bbox_ok": 0,
    "types": {"mechanical": 0, "electronic": 0, "trash": 0},
    "mech": {
        "digits_total": 0,
        "digits_clean": 0,
        "digits_half": 0,
        "digits_sp": 0,
        "ocr_correct": 0,
        "ocr_evaluated": 0,
        "errors": {"blurry": 0, "half": 0, "out": 0, "other": 0},
        "e2e_ok": 0
    },
    "elec": {
        "total": 0,
        "ocr_correct_chars": 0,
        "ocr_evaluated_chars": 0,
        "exact_match": 0,
        "errors": {"blurry": 0, "out": 0, "other": 0},
        "e2e_ok": 0
    },
    "per_img": [],
    "high_conf_errors": []
}

for fname, d in eval_data.items():
    t = d.get("type", "other")
    if t in stats["types"]: stats["types"][t] += 1
    
    if t == "trash": continue
    
    if d.get("obb"): stats["obb_ok"] += 1
    if d.get("digits_bbox_correct"): stats["bbox_ok"] += 1
    
    actual = d.get("actual_text", "").replace(" ", "").replace("_", "")
    img_path = TEST_IMAGES_DIR / fname
    
    pred = ""
    confs = []
    if d.get("digits_bbox_correct") and img_path.exists():
        res = infer_image(img_path, t)
        if res:
            pred, confs = res
            
        # Collect high conf errors
        if len(actual) == len(pred) and t == "mechanical":
            for i, (a, p, c) in enumerate(zip(actual, pred, confs)):
                if a != p and c > 0.60:
                    stats["high_conf_errors"].append({
                        "img": fname,
                        "digit_index": i,
                        "actual_char": a,
                        "pred_char": p,
                        "conf": round(c, 4)
                    })
    
    if t == "mechanical":
        reason = d.get("bbox_miss_reason")
        if not d.get("digits_bbox_correct"):
            if reason == "blurry_occluded": stats["mech"]["errors"]["blurry"] += 1
            elif reason == "half_digit_not_trained": stats["mech"]["errors"]["half"] += 1
            elif reason == "out_of_frame": stats["mech"]["errors"]["out"] += 1
            else: stats["mech"]["errors"]["other"] += 1
            
        details = d.get("digit_details", [])
        n = len(details)
        half = sum(1 for x in details if x.get("is_half"))
        sp = sum(1 for x in details if x.get("is_spurious"))
        clean = n - half - sp
        
        stats["mech"]["digits_total"] += n
        stats["mech"]["digits_clean"] += clean
        stats["mech"]["digits_half"] += half
        stats["mech"]["digits_sp"] += sp
        
        if d.get("digits_bbox_correct") and n > 0:
            stats["per_img"].append({
                "img": fname[:3] + "..." + fname[-6:],
                "actual": actual, "pred": pred, "type": "mech",
                "n": n, "half": half, "sp": sp
            })
            stats["mech"]["ocr_evaluated"] += n
            correct = sum(1 for a, p in zip(actual, pred) if a == p) if len(actual) == len(pred) else 0
            stats["mech"]["ocr_correct"] += correct
            if correct == n and n > 0:
                stats["mech"]["e2e_ok"] += 1
                
    elif t == "electronic":
        stats["elec"]["total"] += 1
        reason = d.get("bbox_miss_reason")
        if not d.get("digits_bbox_correct"):
            if reason == "blurry_occluded": stats["elec"]["errors"]["blurry"] += 1
            elif reason == "out_of_frame": stats["elec"]["errors"]["out"] += 1
            else: stats["elec"]["errors"]["other"] += 1
            
        if d.get("digits_bbox_correct") and actual:
            stats["per_img"].append({
                "img": fname[:3] + "..." + fname[-6:],
                "actual": actual, "pred": pred, "type": "elec",
                "n": len(actual), "half": 0, "sp": 0
            })
            stats["elec"]["ocr_evaluated_chars"] += len(actual)
            # Count correct characters
            correct = sum(1 for a, p in zip(actual, pred) if a == p) if len(actual) == len(pred) else 0
            stats["elec"]["ocr_correct_chars"] += correct
            if actual == pred:
                stats["elec"]["exact_match"] += 1
                stats["elec"]["e2e_ok"] += 1

with open("/home/deist/Downloads/OCR/anh/scratch/full_stats.json", "w") as f:
    json.dump(stats, f)

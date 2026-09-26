import json, cv2, torch, numpy as np, sys, re
from pathlib import Path
from PIL import Image

sys.path.append("/home/deist/Downloads/OCR/anh")
from app import get_yolo_digits, get_yolo_electronic, get_ocr_model, get_parseq_model, _load_model, BASE_DIR, TEST_IMAGES_DIR

eval_file = BASE_DIR / "testocr_eval.json"
with open(eval_file) as f:
    eval_data = json.load(f)

# Statistics
total_images = len(eval_data)
types = {"mechanical": 0, "electronic": 0, "trash": 0, "other": 0}

mech_total = 0
mech_obb_ok = 0
mech_bbox_ok = 0
mech_ocr_ok_digits = 0
mech_total_digits_evaluated = 0

mech_errors_blurry = 0
mech_errors_half = 0
mech_errors_out = 0
mech_errors_other = 0

mech_digit_clean = 0
mech_digit_half = 0
mech_digit_spurious = 0

per_img = []

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

ocr_errors = 0

for fname, d in eval_data.items():
    t = d.get("type", "other")
    if t not in types: types[t] = 0
    types[t] += 1
    
    if t == "mechanical":
        mech_total += 1
        if d.get("obb"): mech_obb_ok += 1
        if d.get("digits_bbox_correct"): mech_bbox_ok += 1
        
        reason = d.get("bbox_miss_reason")
        if reason == "blurry_occluded": mech_errors_blurry += 1
        elif reason == "half_digit_not_trained": mech_errors_half += 1
        elif reason == "out_of_frame": mech_errors_out += 1
        elif reason: mech_errors_other += 1
        
        actual = d.get("actual_text", "").replace(" ", "").replace("_", "")
        img_path = TEST_IMAGES_DIR / fname
        pred = infer_image(img_path, t) if img_path.exists() else ""
        if pred is None: pred = ""
        
        details = d.get("digit_details", [])
        n = len(details)
        half = sum(1 for x in details if x.get("is_half"))
        sp = sum(1 for x in details if x.get("is_spurious"))
        clean = n - half - sp
        mech_total_digits += n
        mech_digit_clean += clean
        mech_digit_half += half
        mech_digit_spurious += sp
        
        if d.get("digits_bbox_correct") and n > 0:
            per_img.append({
                "img": fname[:3] + "..." + fname[-6:],
                "actual": actual,
                "n": n,
                "half": half,
                "sp": sp
            })
            
            mech_total_digits_evaluated += n
            correct_chars = sum(1 for a, p in zip(actual, pred) if a == p) if len(actual) == len(pred) else 0
            if len(actual) == len(pred):
                mech_ocr_ok_digits += correct_chars
                ocr_errors += (n - correct_chars)
            else:
                ocr_errors += n

# Construct the JS arrays and dynamic HTML
import math
obb_pct = mech_obb_ok / mech_total * 100 if mech_total else 0
bbox_pct = mech_bbox_ok / mech_total * 100 if mech_total else 0
ocr_pct = mech_ocr_ok_digits / mech_total_digits_evaluated * 100 if mech_total_digits_evaluated else 0
e2e = (obb_pct/100) * (bbox_pct/100) * (ocr_pct/100) * 100

js_array = "[\n"
for row in per_img:
    js_array += f'  {{img:"{row["img"]}",actual:"{row["actual"]}",n:{row["n"]},half:{row["half"]},sp:{row["sp"]}}},\n'
js_array += "]"

with open("/home/deist/.gemini/antigravity/brain/c327d66a-ae54-4b63-a997-fec47b910363/eval_dashboard.html", "r") as f:
    html = f.read()

# Replace variables using regex
html = re.sub(r'Dựa trên <strong>.*?</strong>', f'Dựa trên <strong>{total_images} ảnh</strong>', html)
html = re.sub(r'<div class="text-3xl font-bold text-\[var\(--primary\)\]">.*?</div>', f'<div class="text-3xl font-bold text-[var(--primary)]">{total_images}</div>', html, count=1)
html = re.sub(r'<div class="text-3xl font-bold" style="color:#22c55e">.*?%</div>', f'<div class="text-3xl font-bold" style="color:#22c55e">{obb_pct:.1f}%</div>', html, count=1)
html = re.sub(r'<div class="text-xs text-\[var\(--muted-foreground\)\]">.*?/.*?</div>', f'<div class="text-xs text-[var(--muted-foreground)]">{mech_obb_ok}/{mech_total}</div>', html, count=1)

# I will just write a clean new template because regex replacement on a complex HTML is extremely brittle

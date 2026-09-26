from pathlib import Path

app_path = Path("/home/deist/Downloads/OCR/anh/app.py")
content = app_path.read_text(encoding="utf-8")

# 1. Update conf in api_testocr_infer to 0.15
content = content.replace("conf=0.25)", "conf=0.15)")

# 2. Add api_testocr_custom_obb endpoint right before api_testocr_infer
custom_obb_code = '''
@app.route("/api/testocr/custom_obb/<name>", methods=["POST"])
def api_testocr_custom_obb(name):
    from flask import request
    data = request.json
    batch = request.args.get("batch") or data.get("batch")
    img_path, _ = find_validate_image_path(name, batch)
    if not img_path:
        img_path = TEST_IMAGES_DIR / name
    if not img_path.is_file():
        return jsonify({"error": "File not found"}), 404
        
    import cv2
    import numpy as np
    import base64
    from PIL import Image
    import torch
    
    img = cv2.imread(str(img_path))
    if img is None:
        return jsonify({"error": "Bad image"}), 400
        
    H, W = img.shape[:2]
    quad = data.get("quad", [])
    if len(quad) != 4:
        return jsonify({"error": "Invalid quad points (need 4 corners)"}), 400
        
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
    
    cls_id = int(data.get("class_id", 0))
    model_bbox = get_yolo_electronic() if cls_id == 1 else get_yolo_digits()
    conf_thresh = 0.25
        
    res_bbox = model_bbox(warped, conf=conf_thresh, iou=0.45, verbose=False)
    boxes = []
    for r in res_bbox:
        for bx in r.boxes:
            boxes.append({
                "x1": float(bx.xyxy[0][0]),
                "y1": float(bx.xyxy[0][1]),
                "x2": float(bx.xyxy[0][2]),
                "y2": float(bx.xyxy[0][3]),
                "conf": float(bx.conf[0].item()),
                "cls": int(bx.cls[0].item())
            })
    boxes.sort(key=lambda b: b["x1"])
    
    digits = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if cls_id == 0:
        ocr_model, ocr_transform = get_ocr_model()
        for i, bx in enumerate(boxes):
            x1, y1, x2, y2 = int(bx["x1"]), int(bx["y1"]), int(bx["x2"]), int(bx["y2"])
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(warped.shape[1], x2); y2 = min(warped.shape[0], y2)
            if x2 <= x1 or y2 <= y1: continue
            
            crop = warped[y1:y2, x1:x2]
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            inp = ocr_transform(crop_pil).unsqueeze(0).to(device)
            pred_str = "?"
            pred_conf = 1.0
            with torch.no_grad():
                preds = ocr_model(inp)
                probs = torch.nn.functional.softmax(preds, dim=1)
                pred_idx = probs.argmax(dim=1).item()
                pred_conf = probs[0, pred_idx].item()
                pred_str = str(pred_idx)
            digits.append({
                "index": i,
                "box": [bx["x1"], bx["y1"], bx["x2"] - bx["x1"], bx["y2"] - bx["y1"]],
                "char": pred_str,
                "conf": bx["conf"],
                "ocr_conf": pred_conf
            })
    else:
        for i, bx in enumerate(boxes):
            digits.append({
                "index": i,
                "box": [bx["x1"], bx["y1"], bx["x2"] - bx["x1"], bx["y2"] - bx["y1"]],
                "char": "?",
                "conf": bx["conf"],
                "ocr_conf": 1.0
            })
            
    _, buf = cv2.imencode(".jpg", warped)
    warped_b64 = base64.b64encode(buf).decode("utf-8")
    
    return jsonify({
        "ok": True,
        "class_id": cls_id,
        "quad": ordered_quad,
        "digits": digits,
        "warped_w": max_w + 2*pad,
        "warped_h": max_h + 2*pad,
        "warped_b64": warped_b64,
        "image_b64": warped_b64,
        "M": M.tolist()
    })

'''

if "@app.route(\"/api/testocr/custom_obb/<name>\"" not in content:
    target_pos = content.find("@app.route(\"/api/testocr/infer/<name>\")")
    content = content[:target_pos] + custom_obb_code + content[target_pos:]
    app_path.write_text(content, encoding="utf-8")
    print("custom_obb endpoint added successfully!")
else:
    print("custom_obb endpoint already exists!")

import sys

with open("app.py", "r") as f:
    content = f.read()

testocr_code = """
@app.route("/testocr")
def view_testocr():
    return render_template("testocr.html")

@app.route("/api/testocr/images")
def api_testocr_images():
    valid = []
    if TEST_IMAGES_DIR.exists():
        for p in TEST_IMAGES_DIR.glob("*.*"):
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                valid.append(p.name)
    return jsonify(sorted(valid))

@app.route("/api/testocr/image/<name>")
def api_testocr_serve(name):
    return send_file(str(TEST_IMAGES_DIR / name))

@app.route("/api/testocr/infer/<name>")
def api_testocr_infer(name):
    img_path = TEST_IMAGES_DIR / name
    if not img_path.is_file():
        return jsonify({"error": "File not found"}), 404
        
    import cv2
    import numpy as np
    img = cv2.imread(str(img_path))
    if img is None:
        return jsonify({"error": "Bad image"}), 400
        
    H, W = img.shape[:2]
    
    # 1. OBB (Screen/Meter Detection)
    obb_model = get_model() # This is yolov8n-obb.pt in app.py
    if not obb_model:
        return jsonify({"error": "OBB model not loaded"}), 503
        
    results = obb_model(img, verbose=False, conf=0.25)
    r = results[0]
    obb = getattr(r, "obb", None)
    if obb is None or len(obb) == 0:
        return jsonify({"error": "No OBB detected"}), 400
        
    best_idx = 0
    best_conf = 0
    for i in range(len(obb)):
        c = float(obb.conf[i].item())
        if c > best_conf:
            best_conf = c
            best_idx = i
            
    quad = [tuple(p) for p in obb.xyxyxyxy[best_idx].tolist()]
    cls_id = int(obb.cls[best_idx].item())
    
    pts = np.array(quad, dtype=np.float32)
    pad = 20
    w1 = int(np.linalg.norm(pts[0] - pts[1]))
    w2 = int(np.linalg.norm(pts[2] - pts[3]))
    h1 = int(np.linalg.norm(pts[1] - pts[2]))
    h2 = int(np.linalg.norm(pts[0] - pts[3]))
    max_w = max(w1, w2)
    max_h = max(h1, h2)
    
    dst = np.array([
        [pad, pad],
        [pad + max_w, pad],
        [pad + max_w, pad + max_h],
        [pad, pad + max_h]
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (max_w + 2*pad, max_h + 2*pad))
    
    if cls_id == 1:
        model_bbox = get_yolo_electronic()
        conf_thresh = 0.5
        iou_thresh = 0.45
    else:
        model_bbox = get_yolo_digits()
        conf_thresh = 0.35
        iou_thresh = 0.45
        
    if model_bbox is None:
        return jsonify({"error": "YOLO bbox model not loaded"}), 503
        
    results_bbox = model_bbox(warped, conf=conf_thresh, iou=iou_thresh, verbose=False)
    boxes = []
    for bx in results_bbox[0].boxes:
        c = int(bx.cls[0].item())
        conf = float(bx.conf[0].item())
        x1, y1, x2, y2 = map(float, bx.xyxy[0].tolist())
        boxes.append({"c": c, "conf": conf, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        
    boxes.sort(key=lambda b: b["x1"])
    
    digits = []
    ocr_model, ocr_transform = get_ocr_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for i, bx in enumerate(boxes):
        if cls_id == 1 or not ocr_model:
            digits.append({
                "index": i,
                "box": [bx["x1"], bx["y1"], bx["x2"] - bx["x1"], bx["y2"] - bx["y1"]],
                "char": "?"
            })
            continue
            
        x1, y1, x2, y2 = int(bx["x1"]), int(bx["y1"]), int(bx["x2"]), int(bx["y2"])
        x1 = max(0, x1); y1 = max(0, y1); x2 = min(warped.shape[1], x2); y2 = min(warped.shape[0], y2)
        if x2 <= x1 or y2 <= y1: continue
        
        crop = warped[y1:y2, x1:x2]
        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        inp = ocr_transform(crop_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            preds = ocr_model(inp)
            pred_idx = preds.argmax(dim=1).item()
            pred_str = str(pred_idx)
            
        digits.append({
            "index": i,
            "box": [bx["x1"], bx["y1"], bx["x2"] - bx["x1"], bx["y2"] - bx["y1"]],
            "char": pred_str,
            "conf": bx["conf"]
        })
        
    return jsonify({
        "ok": True,
        "class_id": cls_id,
        "quad": quad,
        "digits": digits,
        "warped_w": max_w + 2*pad,
        "warped_h": max_h + 2*pad,
        "M": M.tolist()
    })
"""
if "api_testocr_images" not in content:
    content = content.replace('if __name__ == "__main__":', testocr_code + '\nif __name__ == "__main__":')
    with open("app.py", "w") as f:
        f.write(content)

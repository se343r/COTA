import re

with open('/home/deist/Downloads/OCR/anh/app.py', 'r') as f:
    code = f.read()

# Let's extract the part from `def api_testocr_infer(name):` down to the end of the function, and replace it manually.
start_idx = code.find('def api_testocr_infer(name):')
end_idx = code.find('return jsonify({', start_idx)

new_func = """def api_testocr_infer(name):
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
    obb_model = _load_model() # This is yolov8n-obb.pt in app.py
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
    
    pts_sum = [p[0] + p[1] for p in quad]
    pts_diff = [p[0] - p[1] for p in quad]
    tl = quad[np.argmin(pts_sum)]
    br = quad[np.argmax(pts_sum)]
    tr = quad[np.argmax(pts_diff)]
    bl = quad[np.argmin(pts_diff)]
    quad = [tl, tr, br, bl]
    
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
    
    # Get bounding boxes using the digits model
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
    
    # Filter boxes outside the actual OBB region (allowing a margin for edge digits)
    filtered_boxes = []
    margin = pad  # allow the center to be in the padding
    for bx in boxes:
        cx = (bx["x1"] + bx["x2"]) / 2
        cy = (bx["y1"] + bx["y2"]) / 2
        if cx >= pad - margin and cx <= pad + max_w + margin and cy >= pad - margin and cy <= pad + max_h + margin:
            filtered_boxes.append(bx)
    boxes = filtered_boxes

    digits = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    M_inv = np.linalg.inv(M)
    
    ocr_model, ocr_transform = get_ocr_model()
    for i, bx in enumerate(boxes):
        x1_f, y1_f, x2_f, y2_f = bx["x1"], bx["y1"], bx["x2"], bx["y2"]
        cw_pts = np.array([[x1_f, y1_f], [x2_f, y1_f], [x2_f, y2_f], [x1_f, y2_f]], dtype=np.float32).reshape(-1, 1, 2)
        c_orig = cv2.perspectiveTransform(cw_pts, M_inv).reshape(4, 2).tolist()
        
        if not ocr_model:
            digits.append({
                "index": i,
                "box": [bx["x1"], bx["y1"], bx["x2"] - bx["x1"], bx["y2"] - bx["y1"]],
                "box_orig": c_orig,
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
            probs = torch.nn.functional.softmax(preds, dim=1)
            pred_idx = probs.argmax(dim=1).item()
            pred_conf = probs[0, pred_idx].item()
            pred_str = str(pred_idx)
            
        digits.append({
            "index": i,
            "box": [bx["x1"], bx["y1"], bx["x2"] - bx["x1"], bx["y2"] - bx["y1"]],
            "box_orig": c_orig,
            "char": pred_str,
            "conf": bx["conf"],
            "ocr_conf": pred_conf
        })
        
    """

code = code[:start_idx] + new_func + code[end_idx:]

with open('/home/deist/Downloads/OCR/anh/app.py', 'w') as f:
    f.write(code)

from pathlib import Path

app_path = Path("/home/deist/Downloads/OCR/anh/app.py")
content = app_path.read_text(encoding="utf-8")

# Define our new block
old_marker = '@app.route("/testocr")\ndef view_testocr():'

new_code = '''
# -------------------------------------------------------------
# MULTI-BATCH VALIDATION & EDITING PIPELINE SUPPORT
# -------------------------------------------------------------
VALIDATE1_DIR = Path("/home/deist/Downloads/OCR/testocr")
VALIDATE2_DIR = Path("/home/deist/Downloads/OCR/Validate 2")
VALIDATE1_EVAL = BASE_DIR / "validate1_eval.json"
VALIDATE2_EVAL = BASE_DIR / "validate2_eval.json"

def get_validate_info(batch="2"):
    b = str(batch)
    if b == "1":
        return VALIDATE1_DIR, VALIDATE1_EVAL
    return VALIDATE2_DIR, VALIDATE2_EVAL

def find_validate_image_path(name, batch=None):
    if str(batch) == "1":
        p = VALIDATE1_DIR / name
        if p.exists(): return p, "1"
    elif str(batch) == "2":
        p = VALIDATE2_DIR / name
        if p.exists(): return p, "2"
    # Fallback search
    p2 = VALIDATE2_DIR / name
    if p2.exists(): return p2, "2"
    p1 = VALIDATE1_DIR / name
    if p1.exists(): return p1, "1"
    return None, None

@app.route("/validate1")
def view_validate1():
    """Màn hình chỉnh sửa riêng cho tập Validate 1 (48 ảnh) để chuẩn bị train"""
    return render_template("validate1_edit.html")

@app.route("/api/validate1/export_to_train", methods=["POST"])
def api_validate1_export_to_train():
    """Xuất các ảnh đã chỉnh sửa từ validate1_eval.json trực tiếp vào ocr_dataset"""
    import cv2, json
    eval_file = VALIDATE1_EVAL
    if not eval_file.exists():
        return jsonify({"error": "validate1_eval.json không tồn tại"}), 404
        
    with open(eval_file, encoding="utf-8") as f:
        eval_data = json.load(f)
        
    crops_target = BASE_DIR / "ocr_dataset" / "crops"
    labels_target = BASE_DIR / "ocr_dataset" / "labels"
    crops_target.mkdir(parents=True, exist_ok=True)
    labels_target.mkdir(parents=True, exist_ok=True)
    
    exported_images = 0
    clean_digits_count = 0
    half_digits_count = 0
    
    obb_model = _load_model()
    model_bbox = get_yolo_digits()
    
    for fname, d in eval_data.items():
        if d.get("type") != "mechanical" or not d.get("digits_bbox_correct"):
            continue
            
        img_path = VALIDATE1_DIR / fname
        if not img_path.exists(): continue
        
        img = cv2.imread(str(img_path))
        if img is None: continue
        
        # OBB
        res_obb = obb_model(img, conf=0.25, verbose=False)
        if len(res_obb[0].obb) == 0: continue
        best_obb = max(res_obb[0].obb, key=lambda x: x.conf[0].item())
        quad = [tuple(p) for p in best_obb.xyxyxyxy[0].tolist()]
        pts_sum = [p[0] + p[1] for p in quad]
        pts_diff = [p[0] - p[1] for p in quad]
        tl = quad[np.argmin(pts_sum)]
        br = quad[np.argmax(pts_sum)]
        tr = quad[np.argmax(pts_diff)]
        bl = quad[np.argmin(pts_diff)]
        
        pad = 20
        max_w = max(int(np.linalg.norm(np.array(tl) - np.array(tr))), int(np.linalg.norm(np.array(bl) - np.array(br))))
        max_h = max(int(np.linalg.norm(np.array(tl) - np.array(bl))), int(np.linalg.norm(np.array(tr) - np.array(br))))
        pts = np.array([tl, tr, br, bl], dtype=np.float32)
        dst = np.array([[pad, pad], [pad + max_w, pad], [pad + max_w, pad + max_h], [pad, pad + max_h]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(img, M, (max_w + 2*pad, max_h + 2*pad))
        
        W_warp, H_warp = warped.shape[1], warped.shape[0]
        
        boxes = []
        if "boxes" in d and d["boxes"]:
            boxes = d["boxes"]
        else:
            res_bbox = model_bbox(warped, conf=0.35, iou=0.45, verbose=False)
            for bx in res_bbox[0].boxes:
                boxes.append({
                    "x1": float(bx.xyxy[0][0]), "y1": float(bx.xyxy[0][1]),
                    "x2": float(bx.xyxy[0][2]), "y2": float(bx.xyxy[0][3]),
                })
            boxes.sort(key=lambda b: b["x1"])
            
        details = d.get("digit_details", [])
        stem = Path(fname).stem
        
        digit_entries = []
        for idx, bx in enumerate(boxes):
            if idx < len(details):
                det = details[idx]
                if det.get("is_spurious"):
                    continue
                lbl = str(det.get("actual", "")).strip()
                if not lbl or not lbl.isdigit():
                    continue
                is_half = bool(det.get("is_half", False))
                is_dec = bool(det.get("is_decimal", False))
                
                x_norm = bx["x1"] / W_warp
                y_norm = bx["y1"] / H_warp
                w_norm = (bx["x2"] - bx["x1"]) / W_warp
                h_norm = (bx["y2"] - bx["y1"]) / H_warp
                
                digit_entries.append({
                    "digit_index": len(digit_entries),
                    "x": x_norm, "y": y_norm, "w": w_norm, "h": h_norm,
                    "label": lbl,
                    "is_decimal": is_dec,
                    "mid_transition": is_half
                })
                if is_half:
                    half_digits_count += 1
                else:
                    clean_digits_count += 1
                    
        if digit_entries:
            cv2.imwrite(str(crops_target / f"{stem}.jpg"), warped)
            lbl_data = {
                "crop_id": stem,
                "source_image": fname,
                "digits": digit_entries
            }
            with open(labels_target / f"{stem}.json", "w", encoding="utf-8") as f:
                json.dump(lbl_data, f, ensure_ascii=False, indent=2)
            exported_images += 1
            
    return jsonify({
        "ok": True,
        "exported_images": exported_images,
        "clean_digits_count": clean_digits_count,
        "half_digits_count": half_digits_count
    })

@app.route("/testocr")
def view_testocr():
    return render_template("testocr.html")

@app.route("/api/testocr/image/<name>")
def api_testocr_serve(name):
    from flask import request
    batch = request.args.get("batch")
    p, _ = find_validate_image_path(name, batch)
    if p and p.exists():
        return send_file(str(p))
    return jsonify({"error": "File not found"}), 404


@app.route("/api/testocr/evaluate/<name>", methods=["POST"])
def api_testocr_evaluate(name):
    from flask import request
    data = request.json
    batch = request.args.get("batch")
    
    # Determine which file to save to
    if batch:
        _, eval_file = get_validate_info(batch)
    else:
        _, detected_b = find_validate_image_path(name)
        _, eval_file = get_validate_info(detected_b if detected_b else "2")
        
    records = {}
    if eval_file.exists():
        try:
            with open(eval_file, "r", encoding="utf-8") as f:
                records = json.load(f)
        except:
            pass
            
    records[name] = data
    
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
        
    # Also sync with testocr_eval.json if batch 1
    if str(batch) == "1":
        try:
            with open(BASE_DIR / "testocr_eval.json", "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
        except: pass

    return jsonify({"ok": True})

@app.route("/api/testocr/evaluate/<name>", methods=["GET"])
def api_testocr_evaluate_get(name):
    from flask import request
    batch = request.args.get("batch")
    _, eval_file = get_validate_info(batch if batch else "2")
    if not eval_file.exists():
        return jsonify(None)
    import json
    with open(eval_file, "r", encoding="utf-8") as f:
        records = json.load(f)
    return jsonify(records.get(name, None))

@app.route("/api/testocr/all_evals")
def api_testocr_all_evals():
    from flask import request
    batch = request.args.get("batch", "2")
    _, eval_file = get_validate_info(batch)
    if not eval_file.exists():
        return jsonify({})
    import json
    with open(eval_file, "r", encoding="utf-8") as f:
        return jsonify(json.load(f))

@app.route("/api/testocr/images")
def api_testocr_images():
    from flask import request
    batch = request.args.get("batch", "2")
    v_dir, _ = get_validate_info(batch)
    valid = []
    if v_dir.exists():
        for p in v_dir.glob("*.*"):
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                valid.append(p.name)
    return jsonify(sorted(valid))
'''

# Find where api_testocr_infer starts
infer_marker = '@app.route("/api/testocr/infer/<name>")\ndef api_testocr_infer(name):'

old_section = content[content.find(old_marker):content.find(infer_marker)]
content = content.replace(old_section, new_code)

# Now update api_testocr_infer to use find_validate_image_path
old_infer_head = '''@app.route("/api/testocr/infer/<name>")
def api_testocr_infer(name):
    img_path = TEST_IMAGES_DIR / name'''

new_infer_head = '''@app.route("/api/testocr/infer/<name>")
def api_testocr_infer(name):
    from flask import request
    batch = request.args.get("batch")
    img_path, _ = find_validate_image_path(name, batch)
    if not img_path:
        img_path = TEST_IMAGES_DIR / name'''

content = content.replace(old_infer_head, new_infer_head)

app_path.write_text(content, encoding="utf-8")
print("app.py successfully patched!")

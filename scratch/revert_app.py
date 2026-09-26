import re

with open('/home/deist/Downloads/OCR/anh/app.py', 'r') as f:
    code = f.read()

# 1. Revert bbox model loading logic
code = re.sub(
    r'    if cls_id == 1:\s+model_bbox = get_yolo_electronic\(\)\s+else:\s+model_bbox = get_yolo_digits\(\)',
    '    model_bbox = get_yolo_digits()',
    code,
    flags=re.MULTILINE
)

# 2. Revert the big if/else logic block back to just the MECHANICAL part
mechanical_logic = """
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
code = re.sub(
    r'    if cls_id == 1:\s+# ELECTRONIC: Sequence level OCR.*?digits\.append\(\{.*?\}\)\s+else:\s+# MECHANICAL: Use TinyDigitCNN.*?digits\.append\(\{.*?\}\)',
    mechanical_logic,
    code,
    flags=re.DOTALL
)

with open('/home/deist/Downloads/OCR/anh/app.py', 'w') as f:
    f.write(code)


import re

with open('/home/deist/Downloads/OCR/anh/app.py', 'r') as f:
    code = f.read()

# Replace get_parseq_models back to get_parseq_model
old_parseq = """
_parseq_model = None
_parseq_transform = None
_parseq_loaded = False
def get_parseq_model():
    global _parseq_model, _parseq_transform, _parseq_loaded
    if _parseq_loaded: return _parseq_model, _parseq_transform
    _parseq_loaded = True
    device = "cuda" if _device_ok() else "cpu"
    try:
        import torch
        from torchvision import transforms
        path = BASE_DIR / "best_parseq_elec.pt"
        if path.exists():
            _parseq_model = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).to(device).eval()
            _parseq_model.load_state_dict(torch.load(path, map_location=device))
            print(f"Loaded tuned PARSeq from {path}")
            _parseq_transform = transforms.Compose([
                transforms.Resize((32, 128), transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5)
            ])
    except Exception as e:
        print(f"Could not load tuned PARSeq model: {e}")
    return _parseq_model, _parseq_transform
"""

code = re.sub(r'_parseq_model_elec = None.*?return _parseq_model_elec, _parseq_model_mech, _parseq_transform', old_parseq.strip(), code, flags=re.DOTALL)

# In api_testocr_infer
# Replace new logic with old logic
old_infer_logic = """
    if cls_id == 1:
        p_model, p_tf = get_parseq_model()
        pred_str = ""
        pred_conf = 1.0
        for i, bx in enumerate(boxes):
            x1, y1, x2, y2 = int(bx["x1"]), int(bx["y1"]), int(bx["x2"]), int(bx["y2"])
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(warped.shape[1], x2); y2 = min(warped.shape[0], y2)
            
            if x2 > x1 and y2 > y1 and p_model:
                crop = warped[y1:y2, x1:x2]
                crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                inp = p_tf(crop_pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = p_model(inp)
                    pred = logits.softmax(-1)
                    label_pred, prob = p_model.tokenizer.decode(pred)
                    pred_str = label_pred[0]
                    if isinstance(prob, list) and len(prob) > 0 and len(prob[0]) > 0:
                        pred_conf = prob[0].mean().item()
            
            x1_f, y1_f, x2_f, y2_f = bx["x1"], bx["y1"], bx["x2"], bx["y2"]
            cw_pts = np.array([[x1_f, y1_f], [x2_f, y1_f], [x2_f, y2_f], [x1_f, y2_f]], dtype=np.float32).reshape(-1, 1, 2)
            c_orig = cv2.perspectiveTransform(cw_pts, M_inv).reshape(4, 2).tolist()
            
            digits.append({
                "index": i,
                "box": [bx["x1"], bx["y1"], bx["x2"] - bx["x1"], bx["y2"] - bx["y1"]],
                "box_orig": c_orig,
                "char": pred_str,
                "conf": bx["conf"],
                "ocr_conf": pred_conf
            })
    else:
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

code = re.sub(r'p_elec, p_mech, p_tf = get_parseq_models\(\).*?digits\.append\(\{.*?\}\)', old_infer_logic.strip(), code, flags=re.DOTALL)

# In eager loading, fix get_parseq_models back to get_parseq_model
code = code.replace("get_parseq_models()", "get_parseq_model()")

with open('/home/deist/Downloads/OCR/anh/app.py', 'w') as f:
    f.write(code)

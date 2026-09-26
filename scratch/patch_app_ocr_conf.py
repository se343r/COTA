with open("app.py", "r") as f:
    content = f.read()

bad_code = """        with torch.no_grad():
            preds = ocr_model(inp)
            pred_idx = preds.argmax(dim=1).item()
            pred_str = str(pred_idx)
            
        digits.append({
            "index": i,
            "box": [bx["x1"], bx["y1"], bx["x2"] - bx["x1"], bx["y2"] - bx["y1"]],
            "char": pred_str,
            "conf": bx["conf"]
        })"""

good_code = """        with torch.no_grad():
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
        })"""

if "ocr_conf" not in content:
    content = content.replace(bad_code, good_code)
    with open("app.py", "w") as f:
        f.write(content)

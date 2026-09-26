with open("app.py", "r") as f:
    content = f.read()

bad_code = """    for i, bx in enumerate(boxes):
        if cls_id == 1 or not ocr_model:
            digits.append({
                "index": i,
                "box": [bx["x1"], bx["y1"], bx["x2"] - bx["x1"], bx["y2"] - bx["y1"]],
                "char": "?"
            })
            continue"""

good_code = """    M_inv = np.linalg.inv(M)
    for i, bx in enumerate(boxes):
        x1_f, y1_f, x2_f, y2_f = bx["x1"], bx["y1"], bx["x2"], bx["y2"]
        cw_pts = np.array([[x1_f, y1_f], [x2_f, y1_f], [x2_f, y2_f], [x1_f, y2_f]], dtype=np.float32).reshape(-1, 1, 2)
        c_orig = cv2.perspectiveTransform(cw_pts, M_inv).reshape(4, 2).tolist()
        
        if cls_id == 1 or not ocr_model:
            digits.append({
                "index": i,
                "box": [bx["x1"], bx["y1"], bx["x2"] - bx["x1"], bx["y2"] - bx["y1"]],
                "box_orig": c_orig,
                "char": "?"
            })
            continue"""

bad_code2 = """        digits.append({
            "index": i,
            "box": [bx["x1"], bx["y1"], bx["x2"] - bx["x1"], bx["y2"] - bx["y1"]],
            "char": pred_str,
            "conf": bx["conf"],
            "ocr_conf": pred_conf
        })"""

good_code2 = """        digits.append({
            "index": i,
            "box": [bx["x1"], bx["y1"], bx["x2"] - bx["x1"], bx["y2"] - bx["y1"]],
            "box_orig": c_orig,
            "char": pred_str,
            "conf": bx["conf"],
            "ocr_conf": pred_conf
        })"""

if "M_inv" not in content:
    content = content.replace(bad_code, good_code)
    content = content.replace(bad_code2, good_code2)
    with open("app.py", "w") as f:
        f.write(content)

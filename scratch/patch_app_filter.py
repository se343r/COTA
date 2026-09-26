with open("app.py", "r") as f:
    content = f.read()

bad_code = """    boxes.sort(key=lambda b: b["x1"])
    
    digits = []
    ocr_model, ocr_transform = get_ocr_model()"""

good_code = """    boxes.sort(key=lambda b: b["x1"])
    
    # Filter boxes whose centers are outside the actual OBB region (ignoring pad)
    filtered_boxes = []
    for bx in boxes:
        cx = (bx["x1"] + bx["x2"]) / 2
        cy = (bx["y1"] + bx["y2"]) / 2
        if cx >= pad and cx <= pad + max_w and cy >= pad and cy <= pad + max_h:
            filtered_boxes.append(bx)
    boxes = filtered_boxes
    
    digits = []
    ocr_model, ocr_transform = get_ocr_model()"""

if "filtered_boxes = []" not in content:
    content = content.replace(bad_code, good_code)
    with open("app.py", "w") as f:
        f.write(content)

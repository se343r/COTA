import re

with open("app.py", "r") as f:
    content = f.read()

get_model_func = """_yolo_digits = None
def get_yolo_digits():
    global _yolo_digits
    if _yolo_digits is not None:
        return _yolo_digits
    path = BASE_DIR / "yolov8n-digits-bbox.pt"
    if path.exists():
        try:
            from ultralytics import YOLO
            _yolo_digits = YOLO(str(path))
            return _yolo_digits
        except Exception as e:
            print("Could not load YOLO digits model:", e)
            return None
    return None
"""

if "_yolo_digits = None" not in content:
    content = content.replace("def get_model():", get_model_func + "\ndef get_model():")

heuristic_pattern = r'if cls_id == 0 and not digits:.*?(?=return jsonify)'
heuristic_replacement = """if not digits:
            yolo_digits = get_yolo_digits()
            if yolo_digits is not None:
                results = yolo_digits(warped, conf=0.1, verbose=False)
                for r in results:
                    for box in r.boxes:
                        c = int(box.cls[0].item())
                        if (cls_id == 0 and c == 0) or (cls_id == 1 and c == 1):
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            cw = warped.shape[1]
                            ch = warped.shape[0]
                            digits.append({
                                "cx": (x1 + x2) / 2 / cw,
                                "cy": (y1 + y2) / 2 / ch,
                                "w": (x2 - x1) / cw,
                                "h": (y2 - y1) / ch,
                                "label": ""
                            })
                digits.sort(key=lambda d: d["cx"])
        
        """
content = re.sub(heuristic_pattern, heuristic_replacement, content, flags=re.DOTALL)

with open("app.py", "w") as f:
    f.write(content)

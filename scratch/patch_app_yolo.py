import re

with open("app.py", "r") as f:
    content = f.read()

# Add global model loading
if "yolo_digits =" not in content:
    import_block = "from ultralytics import YOLO\n"
    if "from ultralytics import YOLO" not in content:
        content = content.replace("import cv2\n", "import cv2\nfrom ultralytics import YOLO\n")
    
    model_load = 'model = YOLO("yolov8n-obb.pt")\nif Path("yolov8n-digits-bbox.pt").exists():\n    yolo_digits = YOLO("yolov8n-digits-bbox.pt")\nelse:\n    yolo_digits = None\n'
    content = content.replace('model = YOLO("yolov8n-obb.pt")', model_load)


# Replace OpenCV heuristics with YOLO
heuristic_pattern = r'if cls_id == 0 and not digits:.*?(?=return jsonify)'
heuristic_replacement = """if not digits and yolo_digits is not None:
            results = yolo_digits(warped, conf=0.1, verbose=False)
            for r in results:
                for box in r.boxes:
                    c = int(box.cls[0].item())
                    # only keep boxes matching the meter type loosely
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

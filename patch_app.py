import sys

with open("app.py", "r") as f:
    content = f.read()

# 1. Add OCR model loader and TEST_IMAGES_DIR
loader_code = """
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

class TinyDigitCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

_ocr_model = None
_ocr_transform = None

def get_ocr_model():
    global _ocr_model, _ocr_transform
    if _ocr_model is not None:
        return _ocr_model, _ocr_transform
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyDigitCNN(num_classes=10)
    
    path = BASE_DIR / "output_digit_clf" / "best_digit_clf.pt"
    if path.exists():
        try:
            model.load_state_dict(torch.load(str(path), map_location=device))
            model.to(device)
            model.eval()
            _ocr_model = model
            _ocr_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor()
            ])
            print(f"Loaded OCR model from {path}")
        except Exception as e:
            print(f"Failed to load OCR model: {e}")
            _ocr_model = None
            _ocr_transform = None
    return _ocr_model, _ocr_transform

TEST_IMAGES_DIR = Path("/home/deist/Downloads/OCR/testocr")
if not TEST_IMAGES_DIR.exists():
    TEST_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

"""

if "class TinyDigitCNN" not in content:
    content = content.replace("app = Flask(__name__)", loader_code + "\napp = Flask(__name__)")


# 2. Add full pipeline endpoint
pipeline_code = """
@app.route("/api/testocr/images")
def api_testocr_images():
    valid = []
    if TEST_IMAGES_DIR.exists():
        for p in TEST_IMAGES_DIR.glob("*.*"):
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                valid.append(p.name)
    return jsonify(sorted(valid))

@app.route("/testocr/<name>")
def api_testocr_serve(name):
    return send_file(str(TEST_IMAGES_DIR / name))

@app.route("/api/testocr/infer/<name>")
def api_testocr_infer(name):
    # Full pipeline: Image -> OBB -> Crop -> YOLO BBox -> TinyDigitCNN
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
    obb_model = get_yolo_obb()
    if not obb_model:
        return jsonify({"error": "OBB model not loaded"}), 503
        
    # We use yolo_obb.pt
    results = obb_model(img, verbose=False, conf=0.25)
    r = results[0]
    obb = getattr(r, "obb", None)
    if obb is None or len(obb) == 0:
        return jsonify({"error": "No OBB detected"}), 400
        
    # Find best OBB
    best_idx = 0
    best_conf = 0
    for i in range(len(obb)):
        c = float(obb.conf[i].item())
        if c > best_conf:
            best_conf = c
            best_idx = i
            
    quad = [tuple(p) for p in obb.xyxyxyxy[best_idx].tolist()]
    cls_id = int(obb.cls[best_idx].item())
    
    # 2. Rectify Crop
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
    
    # 3. Detect Digits using YOLO
    if cls_id == 1:
        # Electronic
        model_bbox = get_yolo_electronic()
        conf_thresh = 0.5
        iou_thresh = 0.45
    else:
        # Mechanical
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
        
    # Sort boxes left to right
    boxes.sort(key=lambda b: b["x1"])
    
    # 4. OCR
    digits = []
    ocr_model, ocr_transform = get_ocr_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for i, bx in enumerate(boxes):
        if cls_id == 1:
            digits.append({
                "index": i,
                "box": [bx["x1"], bx["y1"], bx["x2"] - bx["x1"], bx["y2"] - bx["y1"]],
                "char": "?"
            })
            continue
            
        if not ocr_model:
            digits.append({
                "index": i,
                "box": [bx["x1"], bx["y1"], bx["x2"] - bx["x1"], bx["y2"] - bx["y1"]],
                "char": "?"
            })
            continue
            
        # Crop from warped image
        x1, y1, x2, y2 = int(bx["x1"]), int(bx["y1"]), int(bx["x2"]), int(bx["y2"])
        # ensure bounds
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
    content = content.replace('if __name__ == "__main__":', pipeline_code + '\nif __name__ == "__main__":')

with open("app.py", "w") as f:
    f.write(content)

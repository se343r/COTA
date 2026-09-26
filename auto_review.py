import json
import math
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import sys

# Try importing shapely for IoU
try:
    from shapely.geometry import Polygon
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shapely"])
    from shapely.geometry import Polygon

BASE_DIR = Path(".")
LABELS_DIR = BASE_DIR / "crops" / "labels"
IMAGES_DIR = BASE_DIR / "images"
REVIEW_STATE_PATH = BASE_DIR / "review_state.json"

def get_best_model():
    for p in [
        BASE_DIR / "runs" / "obb" / "train-2" / "weights" / "best.pt",
        BASE_DIR / "runs" / "obb" / "train" / "weights" / "best.pt"
    ]:
        if p.exists():
            return YOLO(p)
    return None

def parse_label(label_path, W, H):
    lines = [l for l in label_path.read_text().splitlines() if l.strip()]
    if not lines: return None
    toks = lines[0].split()
    if len(toks) != 9: return None
    cls_id = int(toks[0])
    coords = [float(t) for t in toks[1:]]
    corners = [(coords[i] * W, coords[i + 1] * H) for i in range(0, 8, 2)]
    return cls_id, corners

def poly_iou(poly1, poly2):
    p1 = Polygon(poly1)
    p2 = Polygon(poly2)
    if not p1.is_valid: p1 = p1.buffer(0)
    if not p2.is_valid: p2 = p2.buffer(0)
    inter = p1.intersection(p2).area
    union = p1.union(p2).area
    return inter / union if union > 0 else 0

def main():
    model = get_best_model()
    if not model:
        print("No trained model found.")
        return
        
    labels = list(LABELS_DIR.glob("*.txt"))
    print(f"Found {len(labels)} manual labels. Starting AI review...")
    
    flagged = {}
    class_mismatch_count = 0
    low_iou_count = 0
    missed_count = 0
    
    for idx, lp in enumerate(labels):
        stem = lp.stem
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            p = IMAGES_DIR / (stem + ext)
            if p.is_file():
                img_path = p
                break
        if not img_path: continue
        
        try:
            with Image.open(img_path) as im:
                W, H = im.size
        except:
            continue
            
        gt = parse_label(lp, W, H)
        if not gt: continue
        gt_cls, gt_corners = gt
        
        results = model.predict(str(img_path), conf=0.3, verbose=False)
        obb = getattr(results[0], "obb", None)
        
        if obb is None or len(obb) == 0:
            flagged[img_path.name] = "flagged" # Model didn't find anything, worth checking
            missed_count += 1
            print(f"[{idx+1}/{len(labels)}] {img_path.name}: 🚩 MISSED by model")
            continue
            
        # Get top prediction
        pred_cls = int(obb.cls[0].item())
        pred_quad = [tuple(p) for p in obb.xyxyxyxy[0].tolist()]
        
        iou = poly_iou(gt_corners, pred_quad)
        
        is_flagged = False
        reasons = []
        if pred_cls != gt_cls:
            reasons.append(f"Class mismatch (GT:{gt_cls}, Pred:{pred_cls})")
            class_mismatch_count += 1
            is_flagged = True
            
        if iou < 0.85:
            reasons.append(f"Low IoU ({iou:.2f})")
            low_iou_count += 1
            is_flagged = True
            
        if is_flagged:
            flagged[img_path.name] = "flagged"
            print(f"[{idx+1}/{len(labels)}] {img_path.name}: 🚩 {' | '.join(reasons)}")
            
    print(f"\n--- AI REVIEW SUMMARY ---")
    print(f"Total reviewed: {len(labels)}")
    print(f"Flagged for human review: {len(flagged)}")
    print(f"  - Missed by model: {missed_count}")
    print(f"  - Class mismatch: {class_mismatch_count}")
    print(f"  - Low IoU (< 0.85): {low_iou_count}")
    
    # Merge into review_state.json
    state = {}
    if REVIEW_STATE_PATH.exists():
        try:
            state = json.loads(REVIEW_STATE_PATH.read_text())
        except: pass
        
    for name, status in flagged.items():
        # Only overwrite if not already explicitly marked 'ok' by human
        if state.get(name) != "ok":
            state[name] = status
            
    REVIEW_STATE_PATH.write_text(json.dumps(state, indent=1))
    print(f"Saved {len(flagged)} flagged items to review_state.json")
    print("Open the web app -> '🔍 Rà soát nhãn' -> Filter: '🚩 Sai' to see them.")

if __name__ == '__main__':
    main()

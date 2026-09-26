import cv2, numpy as np, json
from pathlib import Path

digits_dir = Path('../crops/digits_data')
out_dir = Path('elec_contours')
out_dir.mkdir(exist_ok=True)

def find_image(name):
    for p in Path('../filtered').rglob(name): return p
    for p in Path('..').rglob(name):
        if 'yolo' not in str(p) and 'scratch' not in str(p): return p
    return None

def warp(img, corners):
    pts = np.array(corners, dtype=np.float32)
    pad = 5
    max_w = max(int(np.linalg.norm(pts[0]-pts[1])), int(np.linalg.norm(pts[2]-pts[3])))
    max_h = max(int(np.linalg.norm(pts[1]-pts[2])), int(np.linalg.norm(pts[0]-pts[3])))
    dst = np.array([[pad,pad],[pad+max_w,pad],[pad+max_w,pad+max_h],[pad,pad+max_h]], np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(img, M, (max_w+2*pad, max_h+2*pad))

count = 0
for jf in digits_dir.glob('*.json'):
    with open(jf) as f: d = json.load(f)
    if d.get('class_id', 0) != 1: continue
    ft = d.get('full_text','').replace('_','').replace('.','').strip()
    if not ft: continue
    
    img_path = find_image(d['image'])
    if not img_path: continue
    img = cv2.imread(str(img_path))
    if img is None: continue
    
    strip = warp(img, d['corners'])
    
    # Try finding contours
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    
    # Increase contrast
    gray = cv2.equalizeHist(gray)
    
    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    
    # Morphological operations to connect segments of the same digit
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if h > strip.shape[0] * 0.4 and w > 10: # reasonable size
            boxes.append((x, y, w, h))
            
    boxes = sorted(boxes, key=lambda b: b[0])
    
    out = strip.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(out, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    cv2.imwrite(str(out_dir / f"{d['image'][:10]}_cnt{len(boxes)}.jpg"), out)
    count += 1
    if count >= 20: break
    
print("Done")

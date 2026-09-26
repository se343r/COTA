import json, cv2, numpy as np
from pathlib import Path

BASE_DIR = Path("/home/deist/Downloads/OCR/anh")
digits_dir = BASE_DIR / 'crops/digits_data'
out_dir = BASE_DIR / 'elec_slices_len6'
out_dir.mkdir(exist_ok=True)

def find_image(name):
    for p in (BASE_DIR / 'filtered').rglob(name): return p
    for p in BASE_DIR.rglob(name):
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
    if not ft or not all(c in '0123456789' for c in ft): continue
    if len(ft) != 6: continue # ONLY LEN 6
    
    img_path = find_image(d['image'])
    if not img_path: continue
    img = cv2.imread(str(img_path))
    if img is None: continue
    
    strip = warp(img, d['corners'])
    h, w = strip.shape[:2]
    
    n = len(ft)
    pad_x = int(w * 0.05)
    usable_w = w - 2*pad_x
    step = usable_w / n
    
    for i, char in enumerate(ft):
        x1 = pad_x + int(i * step)
        x2 = pad_x + int((i+1) * step)
        digit_crop = strip[:, x1:x2]
        
        fname = f"{d['image'][:10]}_{i}_{char}.jpg"
        cv2.imwrite(str(out_dir / fname), digit_crop)
    
    count += 1
print(f"Processed {count} electronic images (len 6), saved slices to {out_dir}")

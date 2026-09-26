import json, cv2, numpy as np, os, random
from pathlib import Path

BASE_DIR = Path("/home/deist/Downloads/OCR/anh")
digits_dir = BASE_DIR / 'crops/digits_data'

out_dir = BASE_DIR / 'parseq_data'

def find_image(name):
    for p in (BASE_DIR / 'filtered').rglob(name): return p
    for p in BASE_DIR.rglob(name):
        if 'yolo' not in str(p) and 'scratch' not in str(p) and 'parseq' not in str(p): return p
    return None

def warp(img, corners):
    pts = np.array(corners, dtype=np.float32)
    pad = 5
    max_w = max(int(np.linalg.norm(pts[0]-pts[1])), int(np.linalg.norm(pts[2]-pts[3])))
    max_h = max(int(np.linalg.norm(pts[1]-pts[2])), int(np.linalg.norm(pts[0]-pts[3])))
    dst = np.array([[pad,pad],[pad+max_w,pad],[pad+max_w,pad+max_h],[pad,pad+max_h]], np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(img, M, (max_w+2*pad, max_h+2*pad))

samples = []
for jf in digits_dir.glob('*.json'):
    with open(jf) as f: d = json.load(f)
    if d.get('class_id', 0) != 1: continue
    ft = d.get('full_text','').replace('_','').replace('.','').strip()
    if not ft or not all(c in '0123456789' for c in ft): continue
    
    img_path = find_image(d['image'])
    if not img_path: continue
    samples.append((img_path, d, ft))

random.seed(42)
random.shuffle(samples)

split = int(len(samples) * 0.85)
train_samples = samples[:split]
val_samples = samples[split:]

def write_split(split_name, split_samples):
    split_dir = out_dir / split_name
    img_dir = split_dir / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)
    gt_file = split_dir / 'gt.txt'
    
    with open(gt_file, 'w', encoding='utf-8') as gf:
        for img_path, d, ft in split_samples:
            img = cv2.imread(str(img_path))
            strip = warp(img, d['corners'])
            fname = d['image']
            cv2.imwrite(str(img_dir / fname), strip)
            gf.write(f"{fname}\t{ft}\n")
    print(f"{split_name}: {len(split_samples)} images")

write_split('train', train_samples)
write_split('val', val_samples)

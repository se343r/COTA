import cv2
import glob
import os
import random
from pathlib import Path

YOLO_DIR = Path("yolo_digits_dataset")
OUT_DIR = Path("/home/deist/.gemini/antigravity/brain/c327d66a-ae54-4b63-a997-fec47b910363/scratch/yolo_vis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

img_paths = glob.glob(str(YOLO_DIR / "images/train/*.jpg"))
random.seed(42)
random.shuffle(img_paths)

selected = img_paths[:10]
md_lines = ["````carousel"]

for i, img_path in enumerate(selected):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    
    stem = Path(img_path).stem
    lbl_path = YOLO_DIR / "labels/train" / f"{stem}.txt"
    
    if lbl_path.exists():
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)
                    
                    color = (0, 255, 0) if cls == 0 else (0, 165, 255)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
    out_path = OUT_DIR / f"vis_{i}.jpg"
    cv2.imwrite(str(out_path), img)
    if i > 0:
        md_lines.append("<!-- slide -->")
    md_lines.append(f"![YOLO Image {i}]({out_path.absolute()})")

md_lines.append("````")

with open("/home/deist/.gemini/antigravity/brain/c327d66a-ae54-4b63-a997-fec47b910363/yolo_vis.md", "w") as f:
    f.write("\n".join(md_lines))

print("Done")

import cv2
from pathlib import Path
val_dir = Path("/home/deist/Downloads/OCR/anh/parseq_mech_data/val/images")
for p in list(val_dir.glob("*.jpg"))[:3]:
    img = cv2.imread(str(p))
    print(f"{p.name}: shape {img.shape}")

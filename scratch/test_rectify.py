import cv2
import numpy as np
from pathlib import Path
import base64

def rectify_obb(img_path, obb_norm_quad):
    img = cv2.imread(str(img_path))
    if img is None: return None
    H, W = img.shape[:2]
    
    # Quad is [x1, y1, x2, y2, x3, y3, x4, y4] in normalized coords
    pts = np.array(obb_norm_quad).reshape(4, 2)
    pts[:, 0] *= W
    pts[:, 1] *= H
    pts = pts.astype(np.float32)
    
    # Compute width and height of the bounding box
    w1 = np.linalg.norm(pts[0] - pts[1])
    w2 = np.linalg.norm(pts[2] - pts[3])
    h1 = np.linalg.norm(pts[1] - pts[2])
    h2 = np.linalg.norm(pts[3] - pts[0])
    
    max_w = max(int(w1), int(w2))
    max_h = max(int(h1), int(h2))
    
    dst_pts = np.array([
        [0, 0],
        [max_w - 1, 0],
        [max_w - 1, max_h - 1],
        [0, max_h - 1]
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (max_w, max_h))
    
    return warped, M, pts.tolist(), max_w, max_h

# Test on one image
lbl = list(Path("crops/labels").glob("*.txt"))[0]
name = lbl.stem + ".jpg"
with open(lbl) as f:
    parts = f.readline().strip().split()
    nums = [float(p) for p in parts[1:9]]

warped, M, pts, w, h = rectify_obb(Path(name), nums)
print("Warped shape:", warped.shape)
print("Matrix:", M.tolist())

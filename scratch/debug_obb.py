import sys, cv2, torch, json
sys.path.append("/home/deist/Downloads/OCR/anh")
from app import _load_model, get_yolo_digits, BASE_DIR, TEST_IMAGES_DIR
from ultralytics import YOLO

img_path = TEST_IMAGES_DIR / "2aoboqyumogfwvc9fiskxwswtdkljso9hh2he6i02.jpg"
img = cv2.imread(str(img_path))
obb_model = _load_model()
res = obb_model(img, conf=0.25, verbose=False)

import numpy as np

for i, bx in enumerate(res[0].obb):
    print(f"OBB {i}: conf={bx.conf[0].item():.3f}, cls={int(bx.cls[0].item())}")
    pts = bx.xyxyxyxy[0].cpu().numpy()
    
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    pts = rect
    
    pad = 5
    max_w = max(int(np.linalg.norm(pts[0]-pts[1])), int(np.linalg.norm(pts[2]-pts[3])))
    max_h = max(int(np.linalg.norm(pts[1]-pts[2])), int(np.linalg.norm(pts[0]-pts[3])))
    dst = np.array([[pad,pad],[pad+max_w,pad],[pad+max_w,pad+max_h],[pad,pad+max_h]], np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (max_w+2*pad, max_h+2*pad))
    
    print(f"Warped shape: {warped.shape}")
    
    model_bbox = get_yolo_digits()
    res_bbox = model_bbox(warped, conf=0.35, verbose=False)
    print(f"Digits found: {len(res_bbox[0].boxes)}")
    for digit in res_bbox[0].boxes:
        print(f"  - digit conf: {digit.conf[0].item():.3f}, box: {digit.xyxy[0].tolist()}")


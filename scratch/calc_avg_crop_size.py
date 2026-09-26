import json, cv2, torch, numpy as np, sys
from pathlib import Path

BASE_DIR = Path("/home/deist/Downloads/OCR/anh")
TEST_IMAGES_DIR = Path("/home/deist/Downloads/OCR/testocr")

sys.path.append(str(BASE_DIR))
from app import get_yolo_digits, _load_model

with open(BASE_DIR / "testocr_eval.json") as f:
    eval_data = json.load(f)

total_w = 0
total_h = 0
count = 0

all_w = []
all_h = []

for fname, d in eval_data.items():
    if d.get("type") == "mechanical" and d.get("digits_bbox_correct"):
        img_path = TEST_IMAGES_DIR / fname
        if not img_path.exists(): continue
        
        img = cv2.imread(str(img_path))
        obb_model = _load_model()
        res_obb = obb_model(img, conf=0.25, verbose=False)
        if len(res_obb[0].obb) == 0: continue
        
        best_obb = max(res_obb[0].obb, key=lambda x: x.conf[0].item())
        pts_raw = best_obb.xyxyxyxy[0].cpu().numpy()
        rect = np.zeros((4, 2), dtype="float32")
        s = pts_raw.sum(axis=1)
        rect[0] = pts_raw[np.argmin(s)]
        rect[2] = pts_raw[np.argmax(s)]
        diff = np.diff(pts_raw, axis=1)
        rect[1] = pts_raw[np.argmin(diff)]
        rect[3] = pts_raw[np.argmax(diff)]
        pts = rect
        pad = 20
        max_w = max(int(np.linalg.norm(pts[0]-pts[1])), int(np.linalg.norm(pts[2]-pts[3])))
        max_h = max(int(np.linalg.norm(pts[1]-pts[2])), int(np.linalg.norm(pts[0]-pts[3])))
        dst = np.array([[pad,pad],[pad+max_w,pad],[pad+max_w,pad+max_h],[pad,pad+max_h]], np.float32)
        M = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(img, M, (max_w+2*pad, max_h+2*pad))
        
        model_bbox = get_yolo_digits()
        res_bbox = model_bbox(warped, conf=0.35, iou=0.45, verbose=False)
        
        for bx in res_bbox[0].boxes:
            x1, y1, x2, y2 = int(bx.xyxy[0][0]), int(bx.xyxy[0][1]), int(bx.xyxy[0][2]), int(bx.xyxy[0][3])
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(warped.shape[1], x2); y2 = min(warped.shape[0], y2)
            if x2 > x1 and y2 > y1:
                w = x2 - x1
                h = y2 - y1
                all_w.append(w)
                all_h.append(h)
                total_w += w
                total_h += h
                count += 1

print(f"Total digits analyzed: {count}")
if count > 0:
    print(f"Average Width: {total_w / count:.2f} px")
    print(f"Average Height: {total_h / count:.2f} px")
    print(f"Min Width: {min(all_w)} px, Max Width: {max(all_w)} px")
    print(f"Min Height: {min(all_h)} px, Max Height: {max(all_h)} px")


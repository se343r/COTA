import sys

with open("app.py", "r") as f:
    content = f.read()

bad_sort = """    quad = [tuple(p) for p in obb.xyxyxyxy[best_idx].tolist()]
    cls_id = int(obb.cls[best_idx].item())
    
    pts = np.array(quad, dtype=np.float32)"""

good_sort = """    quad = [tuple(p) for p in obb.xyxyxyxy[best_idx].tolist()]
    cls_id = int(obb.cls[best_idx].item())
    
    pts_sum = [p[0] + p[1] for p in quad]
    pts_diff = [p[0] - p[1] for p in quad]
    tl = quad[np.argmin(pts_sum)]
    br = quad[np.argmax(pts_sum)]
    tr = quad[np.argmax(pts_diff)]
    bl = quad[np.argmin(pts_diff)]
    quad = [tl, tr, br, bl]
    
    pts = np.array(quad, dtype=np.float32)"""

if "pts_diff" not in content:
    content = content.replace(bad_sort, good_sort)
    with open("app.py", "w") as f:
        f.write(content)

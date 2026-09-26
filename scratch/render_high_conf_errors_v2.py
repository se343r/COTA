import json, cv2, shutil
from pathlib import Path

BASE_DIR = Path("/home/deist/Downloads/OCR/anh")
ARTIFACT_DIR = Path("/home/deist/.gemini/antigravity/brain/c327d66a-ae54-4b63-a997-fec47b910363")
TEST_IMAGES_DIR = Path("/home/deist/Downloads/OCR/testocr")

with open(BASE_DIR / "scratch" / "clean_audit.json") as f:
    audit = json.load(f)

raw_errors = audit.get("high_conf_errors", [])

# Phân loại thành 2 nhóm rõ ràng
spurious_cases = ["2aoboqyunhj9qgc2qy78z0d2xuoomwalsxranwgg28.jpg", "2aoboqyuoznlptzvm5i4dtqgdqwtx11pxmmrwbig48.jpg"]

true_ocr_errors = [e for e in raw_errors if e["img"] not in spurious_cases]
spurious_errors = [e for e in raw_errors if e["img"] in spurious_cases]

md = """# 🔍 Báo Cáo Kiểm Tra Kỹ: Các Ca Đoán Sai Với Độ Tự Tin Cao (>60%)

Qua rà soát từng pixel và từng chữ số dự đoán từ API, có tổng cộng **6 trường hợp OCR thực sự phân loại nhầm** khi tự tin > 60%, và **2 trường hợp bị lệch do BBox bắt thừa số ở mép**.

---

## 🎯 Nhóm 1: Lỗi OCR Thực Sự Của Mô Hình TinyDigit (Conf > 60%)
Đây là các ca BBox cắt đúng vị trí chữ số, nhưng mô hình phân loại đọc sai nhãn với độ tự tin cao:

"""

# Group true errors by image
grouped = {}
for e in true_ocr_errors:
    fn = e["img"]
    grouped.setdefault(fn, []).append(e)

import requests
for fn, errs in grouped.items():
    res = requests.get(f"http://127.0.0.1:5000/api/testocr/infer/{fn}")
    if res.status_code != 200: continue
    data = res.json()
    digits = data.get("digits", [])
    
    img_path = TEST_IMAGES_DIR / fn
    orig_dest = ARTIFACT_DIR / fn
    if img_path.exists() and not orig_dest.exists():
        shutil.copy(img_path, orig_dest)
        
    full_act = errs[0]["full_actual"]
    full_pred = errs[0]["full_pred"]
    
    md += f"### 📷 Ảnh: `{fn}`\n"
    md += f"- **Nhãn thực tế:** `{full_act}` | **Mô hình đoán:** `{full_pred}`\n"
    md += f"![{fn}]({orig_dest.absolute()})\n\n"
    md += "| Vị trí chữ số | Ký tự thật | Dự đoán sai | Độ tự tin (Conf) |\n"
    md += "|---|---|---|---|\n"
    
    for err in errs:
        idx = err["digit_index"]
        conf_pct = err["conf"] * 100
        md += f"| Chữ số #{idx+1} | **`{err['actual']}`** | <span style='color:red;'>**`{err['pred']}`**</span> | **{conf_pct:.1f}%** |\n"
    md += "\n---\n\n"

md += """## ⚠️ Nhóm 2: Lỗi Lệch Vị Trí Do BBox Bắt Thừa Số Ở Mép (Shift-by-One)
Ở 2 ảnh này, mô hình BBox phát hiện thừa 1 box rác ở ngoài cùng mép trái (số `1`). Điều này làm toàn bộ chuỗi số phía sau bị đẩy lùi 1 nấc khi đối chiếu theo index, khiến mô hình bị ghi nhận là "đoán sai hàng loạt với độ tự tin 95-99%":

1. **`...wgg28.jpg`**:
   - **Thực tế:** `230520` (6 số)
   - **Mô hình đoán:** `1230520` (7 số)
   - *Phân tích:* Chữ số `1` ở đầu là box thừa (spurious). Các chữ số phía sau `2 3 0 5 2 0` thực tế mô hình **đọc đúng 100%**, nhưng do lệch index nên bị cắm cờ sai.

2. **`...big48.jpg`**:
   - **Thực tế:** `080442` (6 số)
   - **Mô hình đoán:** `1080442` (7 số)
   - *Phân tích:* Chữ số `1` ở đầu là box thừa. Chuỗi phía sau `0 8 0 4 4 2` thực chất mô hình **đoán đúng hoàn toàn**.
"""

with open(ARTIFACT_DIR / "high_conf_errors.md", "w") as f:
    f.write(md)

print("high_conf_errors.md updated successfully!")

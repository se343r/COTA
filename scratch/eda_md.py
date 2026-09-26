import json
from pathlib import Path
from collections import Counter

BASE_DIR = Path("/home/deist/Downloads/OCR/anh")
DIGITS_DIR = BASE_DIR / "crops" / "digits_data"
FILTER_PATH = BASE_DIR / "filter_progress.json"
OCR_PATH = BASE_DIR / "ocr_labels.json"
METER_PATH = BASE_DIR / "meter_classes.json"

md = []
md.append("# Phân tích Tập dữ liệu OCR Công tơ điện")
md.append("\nBáo cáo tổng hợp số liệu thống kê sau quá trình gán nhãn của người dùng.")

if FILTER_PATH.exists():
    with open(FILTER_PATH) as f:
        filter_data = json.load(f)
    md.append("\n## 1. Trạng thái lọc ảnh (Filtering)")
    c = Counter(v["label"] for v in filter_data.values())
    md.append(f"- **Tổng số ảnh đã lọc:** {len(filter_data)}")
    for k, v in c.most_common():
        md.append(f"- `{k}`: {v} ảnh ({(v/len(filter_data))*100:.1f}%)")

if METER_PATH.exists():
    with open(METER_PATH) as f:
        meter_data = json.load(f)
    md.append("\n## 2. Phân loại Công tơ (Meter Types)")
    c = Counter(meter_data.values())
    md.append(f"- **Cơ (Từng số):** {c.get('co', 0)} ảnh")
    md.append(f"- **Điện tử (Chuỗi số):** {c.get('dientu', 0)} ảnh")

if OCR_PATH.exists():
    with open(OCR_PATH) as f:
        ocr_data = json.load(f)
    md.append("\n## 3. Chỉ số toàn trình (OCR Sequences)")
    md.append(f"- **Tổng số ảnh đã gán chuỗi:** {len(ocr_data)}")
    lens = Counter(len(v) for v in ocr_data.values())
    md.append("\n**Phân bố độ dài chuỗi:**")
    md.append("| Độ dài | Số lượng |")
    md.append("|---|---|")
    for k, v in sorted(lens.items()):
        md.append(f"| {k} ký tự | {v} |")

json_files = list(DIGITS_DIR.glob("*.json"))
md.append("\n## 4. Chi tiết Bounding Boxes (Crops)")
md.append(f"- **Số lượng ảnh có lưu BBox:** {len(json_files)}")

class_ids = Counter()
digit_counts = Counter()
label_counts = Counter()
is_between_count = 0
is_decimal_count = 0
unreadable_count = 0

for jf in json_files:
    with open(jf) as f:
        data = json.load(f)
    
    digits = data.get("digits", [])
    digit_counts[len(digits)] += 1
    
    for d in digits:
        lbl = d.get("label", "")
        if lbl:
            label_counts[lbl] += 1
        if lbl == "unreadable":
            unreadable_count += 1
            
        if d.get("is_between"):
            is_between_count += 1
        if d.get("is_decimal"):
            is_decimal_count += 1

md.append("\n**Phân bố số lượng box trên mỗi ảnh:**")
md.append("| Số BBox | Số lượng ảnh |")
md.append("|---|---|")
for k, v in sorted(digit_counts.items()):
    md.append(f"| {k} box | {v} |")

md.append("\n**Thuộc tính đặc biệt của BBox:**")
md.append(f"- `is_between` (Số bị cuộn nửa): {is_between_count}")
md.append(f"- `is_decimal` (Dấu thập phân): {is_decimal_count}")
md.append(f"- `unreadable` (Không thể đọc): {unreadable_count}")

md.append("\n**Tần suất xuất hiện các chữ số:**")
md.append("| Ký tự | Số lần xuất hiện |")
md.append("|---|---|")
for k, v in label_counts.most_common(12):
    md.append(f"| `{k}` | {v} |")

with open(BASE_DIR / "eda_report.md", "w") as f:
    f.write("\n".join(md))


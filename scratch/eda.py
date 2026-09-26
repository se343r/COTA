import json
import glob
from pathlib import Path
from collections import Counter

BASE_DIR = Path("/home/deist/Downloads/OCR/anh")
DIGITS_DIR = BASE_DIR / "crops" / "digits_data"
FILTER_PATH = BASE_DIR / "filter_labels.json"
OCR_PATH = BASE_DIR / "ocr_labels.json"

# 1. Filter stats
if FILTER_PATH.exists():
    with open(FILTER_PATH) as f:
        filter_data = json.load(f)
    print("=== Filter Status ===")
    c = Counter(v["label"] for v in filter_data.values())
    for k, v in c.most_common():
        print(f"  {k}: {v}")
    print(f"  Total filtered: {len(filter_data)}")
    print()

# 2. OCR Stats
if OCR_PATH.exists():
    with open(OCR_PATH) as f:
        ocr_data = json.load(f)
    print("=== OCR Texts (Full sequences) ===")
    print(f"  Total OCR labels: {len(ocr_data)}")
    lens = Counter(len(v) for v in ocr_data.values())
    print("  Sequence lengths:")
    for k, v in sorted(lens.items()):
        print(f"    {k} chars: {v} images")
    print()

# 3. Digit boxes stats
json_files = list(DIGITS_DIR.glob("*.json"))
print(f"=== Digit Boxes ===")
print(f"  Total cropped images with bounding boxes: {len(json_files)}")

class_ids = Counter()
digit_counts = Counter()
label_counts = Counter()
is_between_count = 0
is_decimal_count = 0
unreadable_count = 0

for jf in json_files:
    with open(jf) as f:
        data = json.load(f)
    cid = data.get("class_id", -1)
    class_ids[cid] += 1
    
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

print("  Meter Types (class_id):")
for k, v in class_ids.most_common():
    name = "Mechanical (0)" if k == 0 else "Electronic (1)" if k == 1 else str(k)
    print(f"    {name}: {v}")

print("  Boxes per image:")
for k, v in sorted(digit_counts.items()):
    print(f"    {k} boxes: {v} images")

print("  Special attributes:")
print(f"    is_between (half digits): {is_between_count}")
print(f"    is_decimal (decimals): {is_decimal_count}")
print(f"    unreadable (mờ/không đọc được): {unreadable_count}")

print("  Top 10 labels:")
for k, v in label_counts.most_common(10):
    print(f"    '{k}': {v}")


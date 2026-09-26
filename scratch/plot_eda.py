import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import Counter

BASE_DIR = Path("/home/deist/Downloads/OCR/anh")
DIGITS_DIR = BASE_DIR / "crops" / "digits_data"
FILTER_PATH = BASE_DIR / "filter_progress.json"
METER_PATH = BASE_DIR / "meter_classes.json"
OUT_DIR = Path("/home/deist/.gemini/antigravity/brain/c327d66a-ae54-4b63-a997-fec47b910363")

# Set up matplotlib style
plt.style.use('ggplot')

# 1. Plot Filter Status (Pie Chart)
if FILTER_PATH.exists():
    with open(FILTER_PATH) as f:
        filter_data = json.load(f)
    c = Counter(v["label"] for v in filter_data.values())
    labels = []
    sizes = []
    for k, v in c.most_common():
        labels.append(f"{k} ({v})")
        sizes.append(v)
        
    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    plt.title("Tỷ lệ phân loại ảnh (Filtering Status)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "filter_pie.png", dpi=150)
    plt.close()

# 2. Plot Meter Types (Bar Chart)
if METER_PATH.exists():
    with open(METER_PATH) as f:
        meter_data = json.load(f)
    c = Counter(meter_data.values())
    labels = ["Cơ (co)", "Điện tử (dientu)"]
    counts = [c.get('co', 0), c.get('dientu', 0)]
    
    plt.figure(figsize=(6, 5))
    bars = plt.bar(labels, counts, color=['#3498db', '#e74c3c'])
    plt.title("Phân loại Công tơ (Meter Types)")
    plt.ylabel("Số lượng ảnh")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 5, int(yval), ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT_DIR / "meter_types.png", dpi=150)
    plt.close()

# 3. Plot Digit Frequencies (Bar Chart)
json_files = list(DIGITS_DIR.glob("*.json"))
label_counts = Counter()
for jf in json_files:
    with open(jf) as f:
        data = json.load(f)
    digits = data.get("digits", [])
    for d in digits:
        lbl = d.get("label", "")
        if lbl and lbl != "unreadable":
            label_counts[lbl] += 1

plt.figure(figsize=(10, 5))
items = label_counts.most_common()
# Filter out non-digits just in case, but let's just take top 10 which should be 0-9
items = sorted([(k, v) for k, v in items if k.isdigit()], key=lambda x: x[0])
labels = [x[0] for x in items]
counts = [x[1] for x in items]

bars = plt.bar(labels, counts, color='#2ecc71')
plt.title("Tần suất xuất hiện các chữ số 0-9")
plt.xlabel("Chữ số")
plt.ylabel("Số lần xuất hiện")
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 2, int(yval), ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / "digit_freq.png", dpi=150)
plt.close()

print("Charts generated successfully.")

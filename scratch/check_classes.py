import json
from pathlib import Path

data_dir = Path("crops/digits_data")
classes = set()
for p in data_dir.glob("*.json"):
    with open(p) as f:
        data = json.load(f)
        for d in data.get("digits", []):
            lbl = d.get("label", "")
            if lbl:
                classes.add(lbl)
print("Classes found:", classes)

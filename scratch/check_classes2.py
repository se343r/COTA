import json
from pathlib import Path

data_dir = Path("crops/digits_data")
classes_co = set()
classes_dt = set()
for p in data_dir.glob("*.json"):
    with open(p) as f:
        data = json.load(f)
        cls_id = data.get("class_id", 0)
        for d in data.get("digits", []):
            if cls_id == 0:
                classes_co.add(d.get("label", ""))
            else:
                classes_dt.add(d.get("label", ""))
print("Mechanical classes:", classes_co)
print("Electronic classes:", classes_dt)

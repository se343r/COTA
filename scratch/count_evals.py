import json
with open('/home/deist/Downloads/OCR/anh/testocr_eval.json') as f:
    d = json.load(f)
print(f"Total evaluated images: {len(d)}")
co = sum(1 for v in d.values() if v.get("meter_class") == "co")
dientu = sum(1 for v in d.values() if v.get("meter_class") == "dientu")
print(f"Mechanical: {co}")
print(f"Electronic: {dientu}")

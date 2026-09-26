import json
with open('/home/deist/Downloads/OCR/anh/testocr_eval.json') as f:
    d = json.load(f)
c = 0
for v in d.values():
    if v.get("actual_text") and v.get("actual_text") != "?":
        c += 1
print(f"Valid actual_text count: {c}")

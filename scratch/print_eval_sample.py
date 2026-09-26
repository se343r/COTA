import json
with open('/home/deist/Downloads/OCR/anh/testocr_eval.json') as f:
    d = json.load(f)
print(json.dumps(list(d.values())[0], indent=2))
print(json.dumps(list(d.values())[-1], indent=2))

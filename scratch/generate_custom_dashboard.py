import json

with open("/home/deist/Downloads/OCR/anh/testocr_eval.json") as f:
    eval_data = json.load(f)

# Calculate totals
total = len(eval_data)
mech_total = 0
elec_total = 0
trash_total = 0

mech_obb_ok = 0
mech_bbox_ok = 0
mech_total_digits = 0
mech_clean_digits = 0
mech_half_digits = 0
mech_spurious = 0

per_img_js = []

for k, v in eval_data.items():
    t = v.get("type")
    if t == "trash":
        trash_total += 1
    elif t == "electronic":
        elec_total += 1
    elif t == "mechanical":
        mech_total += 1
        if v.get("obb"): mech_obb_ok += 1
        if v.get("digits_bbox_correct"): mech_bbox_ok += 1
        
        details = v.get("digit_details", [])
        n = len(details)
        half = sum(1 for d in details if d.get("is_half"))
        sp = sum(1 for d in details if d.get("is_spurious"))
        clean = n - half - sp
        
        mech_total_digits += n
        mech_half_digits += half
        mech_spurious += sp
        mech_clean_digits += clean
        
        actual = v.get("actual_text", "")
        per_img_js.append({
            "img": k[:3] + "..." + k[-6:],
            "actual": actual,
            "n": n,
            "half": half,
            "sp": sp
        })

# Format per_img_js string
js_array = "[\n"
for d in per_img_js:
    js_array += f'  {{img:"{d["img"]}",actual:"{d["actual"]}",n:{d["n"]},half:{d["half"]},sp:{d["sp"]}}},\n'
js_array += "]"

# We need actual model inference for "OCR đọc đúng từng số" and "End-to-end".
# The user's eval_dashboard.html had hardcoded 193/200 (96.5%) and ~63% end-to-end.
# For simplicity, I'll use the values extracted directly from testocr_eval.json where possible.
# Actually, the user already had this dashboard and just wants the format.
# Let's read eval_dashboard.html and just update the js_array, pie chart values, etc.
# Wait, I'll just write the entire HTML out based on eval_dashboard.html but formatted.
with open("/home/deist/.gemini/antigravity/brain/c327d66a-ae54-4b63-a997-fec47b910363/eval_dashboard.html", "r") as f:
    orig_html = f.read()

# Instead of blindly doing replacement, let's write out a fresh file mimicking the structure
# Because I only got truncated 92 lines of it! Let me cat the first 100 lines.

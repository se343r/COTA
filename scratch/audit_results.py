import json, requests

with open("/home/deist/Downloads/OCR/anh/testocr_eval.json") as f:
    eval_data = json.load(f)

total = len(eval_data)
obb_true = 0
bbox_true = 0

exact_matches = 0
mech_exact = 0
elec_exact = 0

high_conf_errors = []
per_image_results = []

# Character-level stats
mech_chars_total = 0
mech_chars_correct = 0
elec_chars_total = 0
elec_chars_correct = 0

for fname, d in eval_data.items():
    if d.get("obb"): obb_true += 1
    if d.get("digits_bbox_correct"): bbox_true += 1
    
    res = requests.get(f"http://127.0.0.1:5000/api/testocr/infer/{fname}")
    if res.status_code != 200: continue
    data = res.json()
    cls_id = data.get("class_id") # 0 = mech, 1 = elec
    digits = data.get("digits", [])
    pred_str = "".join([x["char"] for x in digits])
    
    actual_raw = d.get("actual_text", "").strip()
    actual_clean = actual_raw.replace("_", "").strip()
    
    is_elec = (cls_id == 1)
    
    # Exact match check
    if is_elec:
        # For electronic: handle decimal points cleanly
        norm_actual = actual_clean.replace(".", "")
        norm_pred = pred_str.replace(".", "")
        match = (actual_clean == pred_str or norm_actual == norm_pred)
        if match:
            elec_exact += 1
            exact_matches += 1
            
        # Char accuracy
        if d.get("digits_bbox_correct"):
            elec_chars_total += len(norm_actual)
            # count matching chars
            c = sum(1 for a, p in zip(norm_actual, norm_pred) if a == p)
            elec_chars_correct += c
            
    else:
        match = (actual_clean == pred_str)
        if match:
            mech_exact += 1
            exact_matches += 1
            
        # Char accuracy for mech
        if d.get("digits_bbox_correct"):
            # evaluate against valid chars in actual_raw
            eval_chars = [c for c in actual_raw if c != "_"]
            mech_chars_total += len(eval_chars)
            for idx, a_char in enumerate(actual_raw):
                if a_char != "_":
                    if idx < len(digits) and digits[idx].get("char") == a_char:
                        mech_chars_correct += 1

    # Check High Conf Errors for Mech
    if not is_elec and d.get("digits_bbox_correct"):
        for idx, dg in enumerate(digits):
            p_char = dg.get("char")
            p_conf = dg.get("ocr_conf", 0.0)
            if idx < len(actual_raw):
                a_char = actual_raw[idx]
                if a_char != "_" and a_char != p_char and p_conf > 0.60:
                    high_conf_errors.append({
                        "img": fname,
                        "digit_index": idx,
                        "actual": a_char,
                        "pred": p_char,
                        "conf": round(p_conf, 4),
                        "full_actual": actual_raw,
                        "full_pred": pred_str
                    })

    per_image_results.append({
        "fname": fname,
        "type": "elec" if is_elec else "mech",
        "actual": actual_raw,
        "pred": pred_str,
        "match": match,
        "obb_ok": d.get("obb"),
        "bbox_ok": d.get("digits_bbox_correct")
    })

print(f"=== SUMMARY AUDIT (Dataset N={total}) ===")
print(f"OBB Accuracy: {obb_true}/{total} ({obb_true/total*100:.1f}%)")
print(f"BBox Accuracy: {bbox_true}/{total} ({bbox_true/total*100:.1f}%)")
print(f"Exact String Match: {exact_matches}/{total} ({exact_matches/total*100:.1f}%)")
print(f"  Mechanical exact: {mech_exact}")
print(f"  Electronic exact: {elec_exact}")
print(f"Mechanical Char Accuracy: {mech_chars_correct}/{mech_chars_total} ({mech_chars_correct/mech_chars_total*100:.1f}%)")
print(f"Electronic Char Accuracy: {elec_chars_correct}/{elec_chars_total} ({elec_chars_correct/elec_chars_total*100:.1f}%)")
total_chars = mech_chars_total + elec_chars_total
total_chars_ok = mech_chars_correct + elec_chars_correct
print(f"Overall Char Accuracy: {total_chars_ok}/{total_chars} ({total_chars_ok/total_chars*100:.1f}%)")

print(f"\n=== HIGH CONFIDENCE ERRORS (>60%): {len(high_conf_errors)} ca ===")
for err in high_conf_errors:
    conf_p = err['conf'] * 100
    print(f"- {err['img']} | Vị trí #{err['digit_index']+1}: thật '{err['actual']}' -> đoán '{err['pred']}' (conf {conf_p:.1f}%) [Chuỗi: {err['full_actual']} vs {err['full_pred']}]")

with open("/home/deist/Downloads/OCR/anh/scratch/clean_audit.json", "w") as f:
    json.dump({
        "total": total,
        "obb_true": obb_true,
        "bbox_true": bbox_true,
        "exact_matches": exact_matches,
        "mech_exact": mech_exact,
        "elec_exact": elec_exact,
        "mech_chars_correct": mech_chars_correct,
        "mech_chars_total": mech_chars_total,
        "elec_chars_correct": elec_chars_correct,
        "elec_chars_total": elec_chars_total,
        "high_conf_errors": high_conf_errors,
        "per_image": per_image_results
    }, f, indent=2)

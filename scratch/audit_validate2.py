import json, requests

with open("/home/deist/Downloads/OCR/anh/validate2_eval.json") as f:
    eval_data = json.load(f)

total = len(eval_data)
obb_ok = 0
bbox_ok = 0

exact_matches = 0
mech_exact = 0
elec_exact = 0

mech_chars_total = 0
mech_chars_correct = 0
elec_chars_total = 0
elec_chars_correct = 0

# Track errors
high_conf_errors = []
miss_reasons = {}
per_image_results = []

cm_mech = [[0]*10 for _ in range(10)]

for fname, d in eval_data.items():
    if d.get("obb"): obb_ok += 1
    if d.get("digits_bbox_correct"): bbox_ok += 1
    mr = d.get("bbox_miss_reason")
    if mr:
        miss_reasons[mr] = miss_reasons.get(mr, 0) + 1
        
    res = requests.get(f"http://127.0.0.1:5000/api/testocr/infer/{fname}?batch=2")
    if res.status_code != 200:
        per_image_results.append({
            "fname": fname, "error": "API error", "actual": d.get("actual_text", ""), "pred": ""
        })
        continue
        
    data = res.json()
    cls_id = data.get("class_id") # 0 = mech, 1 = elec
    digits = data.get("digits", [])
    pred_str = "".join([x["char"] for x in digits])
    
    actual_raw = d.get("actual_text", "").strip()
    actual_clean = actual_raw.replace("_", "").strip()
    
    is_elec = (cls_id == 1)
    
    # Exact match check
    if is_elec:
        norm_actual = actual_clean.replace(".", "")
        norm_pred = pred_str.replace(".", "")
        match = (actual_clean == pred_str or (norm_actual and norm_actual == norm_pred))
        if match:
            elec_exact += 1
            exact_matches += 1
            
        if d.get("digits_bbox_correct") and norm_actual:
            elec_chars_total += len(norm_actual)
            c = sum(1 for a, p in zip(norm_actual, norm_pred) if a == p)
            elec_chars_correct += c
            
    else:
        match = (actual_clean == pred_str and actual_clean != "")
        if match:
            mech_exact += 1
            exact_matches += 1
            
        if d.get("digits_bbox_correct") and actual_raw:
            eval_chars = [c for c in actual_raw if c != "_"]
            mech_chars_total += len(eval_chars)
            for idx, a_char in enumerate(actual_raw):
                if a_char != "_" and a_char.isdigit():
                    act_digit = int(a_char)
                    if idx < len(digits):
                        p_char = digits[idx].get("char", "")
                        p_conf = digits[idx].get("ocr_conf", 0.0)
                        if p_char.isdigit():
                            p_digit = int(p_char)
                            cm_mech[act_digit][p_digit] += 1
                            if act_digit == p_digit:
                                mech_chars_correct += 1
                            elif p_conf > 0.60:
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
        "eval_type": d.get("type"),
        "actual": actual_raw,
        "pred": pred_str,
        "match": match,
        "obb_ok": d.get("obb"),
        "bbox_ok": d.get("digits_bbox_correct"),
        "miss_reason": d.get("bbox_miss_reason")
    })

print(f"=== KẾT QUẢ THỐNG KÊ TẬP VALIDATE 2 (N={total}) ===")
print(f"OBB Tìm màn hình: {obb_ok}/{total} ({obb_ok/total*100:.1f}%)")
print(f"BBox Bắt đủ số: {bbox_ok}/{total} ({bbox_ok/total*100:.1f}%)")
print(f"Exact String Match (Đúng 100%): {exact_matches}/{total} ({exact_matches/total*100:.1f}%)")
print(f"  - Cơ đúng hoàn toàn: {mech_exact}")
print(f"  - Điện tử đúng hoàn toàn: {elec_exact}")
print(f"Độ chính xác ký tự Cơ: {mech_chars_correct}/{mech_chars_total} ({mech_chars_correct/max(1,mech_chars_total)*100:.1f}%)")
print(f"Độ chính xác ký tự Điện tử: {elec_chars_correct}/{elec_chars_total} ({elec_chars_correct/max(1,elec_chars_total)*100:.1f}%)")
tot_chars = mech_chars_total + elec_chars_total
tot_corr = mech_chars_correct + elec_chars_correct
print(f"Độ chính xác ký tự Toàn bộ (OCR): {tot_corr}/{tot_chars} ({tot_corr/max(1,tot_chars)*100:.1f}%)")

print("\n--- Lý do BBox bị thiếu/lỗi ---")
for r, c in miss_reasons.items():
    print(f"  {r}: {c} ảnh")

print(f"\n--- Các ca tự tin cao (>60%) nhưng đoán sai: {len(high_conf_errors)} ca ---")
for e in high_conf_errors:
    print(f"  {e['img'][:20]}... | #{e['digit_index']+1}: thật '{e['actual']}' -> đoán '{e['pred']}' ({e['conf']*100:.1f}%) | chuỗi: {e['full_actual']} vs {e['full_pred']}")

with open("/home/deist/Downloads/OCR/anh/scratch/val2_stats.json", "w", encoding="utf-8") as f:
    json.dump({
        "total": total,
        "obb_ok": obb_ok,
        "bbox_ok": bbox_ok,
        "exact_matches": exact_matches,
        "mech_exact": mech_exact,
        "elec_exact": elec_exact,
        "mech_chars_correct": mech_chars_correct,
        "mech_chars_total": mech_chars_total,
        "elec_chars_correct": elec_chars_correct,
        "elec_chars_total": elec_chars_total,
        "tot_chars": tot_chars,
        "tot_corr": tot_corr,
        "miss_reasons": miss_reasons,
        "high_conf_errors": high_conf_errors,
        "cm_mech": cm_mech,
        "per_image": per_image_results
    }, f, ensure_ascii=False, indent=2)

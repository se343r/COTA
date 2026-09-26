with open("templates/testocr.html", "r") as f:
    html = f.read()

bad_eval = """        <div style="margin-bottom: 10px;">
            <b style="color: #37c978">🎯 Sửa lỗi OCR (đánh giá từng số):</b>
            <input type="text" id="eval-actual" style="margin-left:10px; width:150px; padding:4px; font-weight:bold; font-size:16px; text-align:center; letter-spacing:4px;" placeholder="Ví dụ: 222482">
            <span style="color:#888; font-size:12px; margin-left:10px">(Chỉ cần sửa nếu máy đọc sai)</span>
            <label style="margin-left:20px; color:#ffb340; font-weight:bold;"><input type="checkbox" id="eval-half-digit"> ⚠️ Có số bị nửa vời (Half digit / Chưa train)</label>
        </div>"""

good_eval = """        <div style="margin-bottom: 10px;">
            <b style="color: #37c978">🎯 Chấm điểm chi tiết từng chữ số (OCR & Half-digit):</b>
            <div id="digit-eval-container" style="display:flex; gap:10px; margin-top:10px; flex-wrap:wrap;"></div>
        </div>"""

html = html.replace(bad_eval, good_eval)

# Now we must patch the JS part where the DOM is reset.
# I need to find the `try {` block in `runInference`
bad_js_reset = """        document.getElementById("eval-actual").value = finalStr;
        document.getElementById("eval-half-digit").checked = false;"""

good_js_reset = """        let digitsHtml = "";
        res.digits.forEach((d, i) => {
            digitsHtml += `<div style="background:#2a333f; padding:8px; border-radius:6px; text-align:center; border:1px solid #3c495a;">
                <div style="font-size:12px; color:#8b98a8;">Dự đoán</div>
                <div style="font-size:24px; color:#4da3ff; font-weight:bold; margin-bottom:5px;">${d.char}</div>
                <div>
                    <input type="text" id="actual-digit-${i}" value="${d.char}" maxlength="1" style="width:30px; padding:4px; text-align:center; font-weight:bold; background:#0f1216; color:white; border:1px solid #4da3ff; border-radius:4px;" title="Thực tế">
                </div>
                <div style="margin-top:5px; font-size:11px; color:#ffb340;">
                    <label><input type="checkbox" id="half-digit-${i}"> Half-digit</label>
                </div>
            </div>`;
        });
        document.getElementById("digit-eval-container").innerHTML = digitsHtml;"""

html = html.replace(bad_js_reset, good_js_reset)

# Patch saveEvaluation variables
bad_save_vars = """    const actual_text = document.getElementById("eval-actual").value;
    const note = document.getElementById("eval-note").value;
    const has_half_digit = document.getElementById("eval-half-digit").checked;"""

good_save_vars = """    const note = document.getElementById("eval-note").value;
    const eval_digits = [];
    let actual_text = "";
    document.querySelectorAll('[id^="actual-digit-"]').forEach((el, i) => {
        let actual = el.value;
        let is_half = document.getElementById("half-digit-" + i).checked;
        actual_text += actual;
        eval_digits.push({ actual: actual, is_half: is_half });
    });"""

html = html.replace(bad_save_vars, good_save_vars)

# Patch JSON payload
bad_payload = """body: JSON.stringify({ type, obb, digits, actual_text, predicted_text: finalStr, has_half_digit, note })"""
# Wait, finalStr is not defined in saveEvaluation! I can read it from the DOM or simply skip it.
# Actually, I can reconstruct predicted_text by mapping res.digits. But `res` is not in scope of saveEvaluation.
# The simplest is to extract predicted_text from `eval_digits` if we pass predicted inside the HTML.
# But `digits` is already `const digits = document.getElementById("eval-digits").checked;` which is a boolean!
# Wait! In `saveEvaluation`, `const digits` is the checkbox "Cắt đúng, đủ số lượng".
# So I'll rename `eval_digits` array to `digit_details` to avoid conflict.

good_save_vars2 = """    const note = document.getElementById("eval-note").value;
    const digit_details = [];
    let actual_text = "";
    document.querySelectorAll('[id^="actual-digit-"]').forEach((el, i) => {
        let actual = el.value;
        let is_half = document.getElementById("half-digit-" + i).checked;
        actual_text += actual;
        digit_details.push({ actual: actual, is_half: is_half });
    });"""
html = html.replace(good_save_vars, good_save_vars2)

bad_payload = """body: JSON.stringify({ type, obb, digits, actual_text, predicted_text: finalStr, has_half_digit, note })"""
good_payload = """body: JSON.stringify({ type, obb, digits_bbox_correct: digits, actual_text, digit_details, note })"""

html = html.replace(bad_payload, good_payload)

with open("templates/testocr.html", "w") as f:
    f.write(html)

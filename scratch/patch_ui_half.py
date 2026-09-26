with open("templates/testocr.html", "r") as f:
    html = f.read()

bad_css = ".sidebar { width:250px; background:var(--panel); border-right:1px solid var(--line); overflow-y:auto; padding:10px; display:flex; flex-direction:column; gap:5px; }"
good_css = ".sidebar { width:350px; background:var(--panel); border-right:1px solid var(--line); overflow-y:auto; padding:10px; display:flex; flex-direction:column; gap:5px; }"

bad_eval = """        <div style="margin-bottom: 10px;">
            <b style="color: #37c978">🎯 Sửa lỗi OCR (đánh giá từng số):</b>
            <input type="text" id="eval-actual" style="margin-left:10px; width:150px; padding:4px; font-weight:bold; font-size:16px; text-align:center; letter-spacing:4px;" placeholder="Ví dụ: 222482">
            <span style="color:#888; font-size:12px; margin-left:10px">(Chỉ cần sửa nếu máy đọc sai)</span>
        </div>"""

good_eval = """        <div style="margin-bottom: 10px;">
            <b style="color: #37c978">🎯 Sửa lỗi OCR (đánh giá từng số):</b>
            <input type="text" id="eval-actual" style="margin-left:10px; width:150px; padding:4px; font-weight:bold; font-size:16px; text-align:center; letter-spacing:4px;" placeholder="Ví dụ: 222482">
            <span style="color:#888; font-size:12px; margin-left:10px">(Chỉ cần sửa nếu máy đọc sai)</span>
            <label style="margin-left:20px; color:#ffb340; font-weight:bold;"><input type="checkbox" id="eval-half-digit"> ⚠️ Có số bị nửa vời (Half digit / Chưa train)</label>
        </div>"""

bad_js_vars = """    const actual_text = document.getElementById("eval-actual").value;
    const note = document.getElementById("eval-note").value;"""

good_js_vars = """    const actual_text = document.getElementById("eval-actual").value;
    const note = document.getElementById("eval-note").value;
    const has_half_digit = document.getElementById("eval-half-digit").checked;"""

bad_js_body = """body: JSON.stringify({ type, obb, digits, actual_text, predicted_text: finalStr, note })"""
good_js_body = """body: JSON.stringify({ type, obb, digits, actual_text, predicted_text: finalStr, has_half_digit, note })"""

bad_js_reset = """        document.getElementById("eval-actual").value = finalStr;"""
good_js_reset = """        document.getElementById("eval-actual").value = finalStr;
        document.getElementById("eval-half-digit").checked = false;"""


html = html.replace(bad_css, good_css)
if "eval-half-digit" not in html:
    html = html.replace(bad_eval, good_eval)
    html = html.replace(bad_js_vars, good_js_vars)
    html = html.replace(bad_js_body, good_js_body)
    html = html.replace(bad_js_reset, good_js_reset)
    
with open("templates/testocr.html", "w") as f:
    f.write(html)

with open("templates/testocr.html", "r") as f:
    html = f.read()

bad_eval = """        <div style="margin-bottom: 10px;">
            <b style="color: #4da3ff">✅ Đánh giá Model:</b>
            <label style="margin-left:10px"><input type="checkbox" id="eval-obb" checked> 1. Tìm đúng màn hình (OBB)</label>
            <label style="margin-left:10px"><input type="checkbox" id="eval-digits" checked> 2. Cắt đúng từng số (BBox)</label>
            <label style="margin-left:10px"><input type="checkbox" id="eval-ocr" checked> 3. OCR đọc chuẩn (100%)</label>
        </div>"""

good_eval = """        <div style="margin-bottom: 10px;">
            <b style="color: #4da3ff">✅ Đánh giá Model Detect:</b>
            <label style="margin-left:10px"><input type="checkbox" id="eval-obb" checked> 1. Tìm đúng màn hình (OBB)</label>
            <label style="margin-left:10px"><input type="checkbox" id="eval-digits" checked> 2. Cắt đúng, đủ số lượng (BBox Digit)</label>
        </div>
        <div style="margin-bottom: 10px;">
            <b style="color: #37c978">🎯 Sửa lỗi OCR (đánh giá từng số):</b>
            <input type="text" id="eval-actual" style="margin-left:10px; width:150px; padding:4px; font-weight:bold; font-size:16px; text-align:center; letter-spacing:4px;" placeholder="Ví dụ: 222482">
            <span style="color:#888; font-size:12px; margin-left:10px">(Chỉ cần sửa nếu máy đọc sai)</span>
        </div>"""

if "eval-actual" not in html:
    html = html.replace(bad_eval, good_eval)
    
    # Also patch JS
    bad_js = """    const digits = document.getElementById("eval-digits").checked;
    const ocr = document.getElementById("eval-ocr").checked;
    const note = document.getElementById("eval-note").value;"""
    
    good_js = """    const digits = document.getElementById("eval-digits").checked;
    const actual_text = document.getElementById("eval-actual").value;
    const note = document.getElementById("eval-note").value;"""
    
    html = html.replace(bad_js, good_js)
    
    bad_js2 = """body: JSON.stringify({ type, obb, digits, ocr, note })"""
    good_js2 = """body: JSON.stringify({ type, obb, digits, actual_text, predicted_text: finalStr, note })"""
    
    html = html.replace(bad_js2, good_js2)
    
    bad_js3 = """document.getElementById("eval-panel").style.display = "block";"""
    good_js3 = """document.getElementById("eval-panel").style.display = "block";
        document.getElementById("eval-actual").value = finalStr;"""
        
    html = html.replace(bad_js3, good_js3)
    
    with open("templates/testocr.html", "w") as f:
        f.write(html)

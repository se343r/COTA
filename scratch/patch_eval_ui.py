with open("templates/testocr.html", "r") as f:
    html = f.read()

bad_html = """    <div class="result-panel" id="result-text" style="color:var(--accent)">-</div>
    <div class="canvas-wrap">"""

good_html = """    <div class="result-panel" id="result-text" style="color:var(--accent)">-</div>
    <div id="eval-panel" style="margin-top: 15px; padding: 15px; background: #1e252e; border: 1px solid var(--line); border-radius: 6px; display: none;">
        <div style="margin-bottom: 10px;">
            <b style="color: #4da3ff">🏷️ Phân loại ảnh thực tế:</b> 
            <label style="margin-left:10px"><input type="radio" name="eval-type" value="mechanical" checked> ⚙️ Công tơ cơ</label>
            <label style="margin-left:10px"><input type="radio" name="eval-type" value="electronic"> ⚡ Công tơ điện tử</label>
            <label style="margin-left:10px"><input type="radio" name="eval-type" value="other"> ❌ Rác / Mờ / Bỏ qua</label>
        </div>
        <div style="margin-bottom: 10px;">
            <b style="color: #4da3ff">✅ Đánh giá Model:</b>
            <label style="margin-left:10px"><input type="checkbox" id="eval-obb" checked> 1. Tìm đúng màn hình (OBB)</label>
            <label style="margin-left:10px"><input type="checkbox" id="eval-digits" checked> 2. Cắt đúng từng số (BBox)</label>
            <label style="margin-left:10px"><input type="checkbox" id="eval-ocr" checked> 3. OCR đọc chuẩn (100%)</label>
        </div>
        <div style="display: flex; align-items: center; gap: 10px;">
            <b>📝 Ghi chú:</b> <input type="text" id="eval-note" style="flex:1; padding:6px; background:#0f1216; color:white; border:1px solid #2a333f; border-radius:4px;" placeholder="Ví dụ: số 6 bị mờ, góc chụp chói sáng...">
            <button onclick="saveEvaluation()" style="background:#4da3ff; color:#0f1216;">💾 Lưu đánh giá (Enter)</button>
            <span id="eval-status" style="color: #37c978; font-weight:bold; min-width:120px;"></span>
        </div>
    </div>
    <div class="canvas-wrap" style="margin-top:15px;">"""

bad_js = """        document.getElementById("result-text").innerHTML = name + " - <span style='color:#37c978'>Kết quả: " + finalStr + "</span> (Class: " + res.class_id + ")" + detailsHtml;
        
    } catch(e) {"""

good_js = """        document.getElementById("result-text").innerHTML = name + " - <span style='color:#37c978'>Kết quả: " + finalStr + "</span> (Class: " + res.class_id + ")" + detailsHtml;
        document.getElementById("eval-panel").style.display = "block";
        document.getElementById("eval-status").innerText = "";
    } catch(e) {"""

if "eval-panel" not in html:
    html = html.replace(bad_html, good_html)
    html = html.replace(bad_js, good_js)
    with open("templates/testocr.html", "w") as f:
        f.write(html)

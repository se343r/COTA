with open("templates/testocr.html", "r") as f:
    html = f.read()

bad_js = """        let finalStr = "";
        res.digits.forEach(d => {
            finalStr += d.char;
        });
        
        document.getElementById("result-text").innerHTML = name + " - <span style='color:#37c978'>Kết quả: " + finalStr + "</span> (Class: " + res.class_id + ")";"""

good_js = """        let finalStr = "";
        let detailsHtml = "<div style='font-size:14px; margin-top:8px; font-weight:normal;'>";
        res.digits.forEach(d => {
            finalStr += d.char;
            if (d.ocr_conf !== undefined) {
                let boxConf = (d.conf * 100).toFixed(1);
                let ocrConf = (d.ocr_conf * 100).toFixed(1);
                detailsHtml += `<span style='display:inline-block; background:#1e252e; padding:4px 8px; margin:2px 4px; border-radius:4px; border:1px solid #2a333f;'>
                    <b style='color:#4da3ff; font-size:16px'>${d.char}</b>
                    <span style='color:#8b98a8; font-size:11px; margin-left:6px'>Box: ${boxConf}% | OCR: ${ocrConf}%</span>
                </span>`;
            }
        });
        detailsHtml += "</div>";
        
        document.getElementById("result-text").innerHTML = name + " - <span style='color:#37c978'>Kết quả: " + finalStr + "</span> (Class: " + res.class_id + ")" + detailsHtml;"""

if "detailsHtml" not in html:
    html = html.replace(bad_js, good_js)
    with open("templates/testocr.html", "w") as f:
        f.write(html)

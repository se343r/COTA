with open("templates/testocr.html", "r") as f:
    html = f.read()

bad_js = """function prevImage() { selectImage(currentIdx - 1); }
function nextImage() { selectImage(currentIdx + 1); }"""

good_js = """function prevImage() { selectImage(currentIdx - 1); }
function nextImage() { selectImage(currentIdx + 1); }

async function saveEvaluation() {
    if (currentIdx < 0) return;
    const name = images[currentIdx];
    const type = document.querySelector('input[name="eval-type"]:checked').value;
    const obb = document.getElementById("eval-obb").checked;
    const digits = document.getElementById("eval-digits").checked;
    const ocr = document.getElementById("eval-ocr").checked;
    const note = document.getElementById("eval-note").value;
    
    document.getElementById("eval-status").innerText = "Đang lưu...";
    
    try {
        const res = await fetch("/api/testocr/evaluate/" + encodeURIComponent(name), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ type, obb, digits, ocr, note })
        }).then(r => r.json());
        
        if (res.ok) {
            document.getElementById("eval-status").innerText = "✔️ Đã lưu!";
            setTimeout(() => nextImage(), 500); // Tự động nhảy sang ảnh tiếp theo
        } else {
            document.getElementById("eval-status").innerText = "❌ Lỗi: " + res.error;
        }
    } catch(e) {
        document.getElementById("eval-status").innerText = "❌ Lỗi kết nối";
    }
}
"""

if "saveEvaluation()" not in html:
    html = html.replace(bad_js, good_js)
    with open("templates/testocr.html", "w") as f:
        f.write(html)

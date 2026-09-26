with open("templates/testocr.html", "r") as f:
    html = f.read()

save_func = """
async function saveEvaluation() {
    if (currentIdx < 0) return;
    const name = images[currentIdx];
    const type = document.querySelector('input[name="eval-type"]:checked').value;
    const obb = document.getElementById("eval-obb").checked;
    const digits_bbox_correct = document.getElementById("eval-digits").checked;
    const note = document.getElementById("eval-note").value;
    
    const digit_details = [];
    let actual_text = "";
    document.querySelectorAll('[id^="actual-digit-"]').forEach((el, i) => {
        let actual = el.value;
        let is_half = document.getElementById("half-digit-" + i).checked;
        actual_text += actual;
        digit_details.push({ actual: actual, is_half: is_half });
    });
    
    document.getElementById("eval-status").innerText = "Đang lưu...";
    
    try {
        const res = await fetch("/api/testocr/evaluate/" + encodeURIComponent(name), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ type, obb, digits_bbox_correct, actual_text, digit_details, note })
        }).then(r => r.json());
        
        if (res.ok) {
            document.getElementById("eval-status").innerText = "✔️ Đã lưu!";
            setTimeout(() => nextImage(), 500);
        } else {
            document.getElementById("eval-status").innerText = "❌ Lỗi: " + res.error;
        }
    } catch(e) {
        document.getElementById("eval-status").innerText = "❌ Lỗi kết nối";
    }
}
"""

if "async function saveEvaluation()" not in html:
    html = html.replace("loadList();\n</script>", save_func + "\nloadList();\n</script>")
    with open("templates/testocr.html", "w") as f:
        f.write(html)

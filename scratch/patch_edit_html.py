from pathlib import Path

html_path = Path("/home/deist/Downloads/OCR/anh/templates/validate1_edit.html")
content = html_path.read_text(encoding="utf-8")

# Add autoDetectCurrent function before init()
func_code = '''
async function autoDetectCurrent() {
  const name = allImages[currentIdx];
  document.getElementById('coords-status').textContent = 'Đang chạy lại detection...';
  const res = await fetch(`/api/testocr/infer/${encodeURIComponent(name)}?batch=1`).then(r => r.json());
  if (res.digits && res.digits.length > 0) {
    currentBoxes = res.digits.map((d, i) => ({
      x: Math.round(d.box[0]),
      y: Math.round(d.box[1]),
      w: Math.round(d.box[2]),
      h: Math.round(d.box[3]),
      actual: d.char,
      is_half: false,
      is_spurious: false,
      is_decimal: false
    }));
    sortBoxes();
    selectedBoxIdx = -1;
    renderDigitCards();
    renderCanvas();
    updateThumbnails();
    document.getElementById('coords-status').textContent = `Đã phát hiện ${currentBoxes.length} hộp số tự động từ model.`;
  }
}
'''

if 'async function autoDetectCurrent()' not in content:
    content = content.replace('init();', func_code + '\ninit();')
    html_path.write_text(content, encoding="utf-8")
    print("autoDetectCurrent added!")
else:
    print("autoDetectCurrent already exists!")

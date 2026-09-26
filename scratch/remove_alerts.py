from pathlib import Path

html_path = Path("/home/deist/Downloads/OCR/anh/templates/validate1_edit.html")
content = html_path.read_text(encoding="utf-8")

# 1. Add toast HTML before </body>
toast_html = '''
<!-- Toast Notification -->
<div id="toast" style="position:fixed; bottom:24px; right:24px; background:#238636; color:#fff; padding:9px 18px; border-radius:6px; font-size:13px; font-weight:600; box-shadow:0 6px 20px rgba(0,0,0,0.6); opacity:0; transition:all 0.25s ease; transform:translateY(12px); pointer-events:none; z-index:9999; display:flex; align-items:center; gap:8px; border:1px solid rgba(255,255,255,0.15);">
  <span id="toast-icon" style="font-size:15px;">✔️</span>
  <span id="toast-msg">Đã lưu thành công!</span>
</div>
'''

content = content.replace('</body>', toast_html + '\n</body>')

# 2. Add showToast JS function
toast_js = '''
function showToast(msg, isError = false) {
  const t = document.getElementById('toast');
  const m = document.getElementById('toast-msg');
  const ic = document.getElementById('toast-icon');
  if (!t) return;
  m.textContent = msg;
  ic.textContent = isError ? '❌' : '✔️';
  t.style.background = isError ? '#da3633' : '#238636';
  t.style.opacity = '1';
  t.style.transform = 'translateY(0)';
  clearTimeout(t._timer);
  t._timer = setTimeout(() => {
    t.style.opacity = '0';
    t.style.transform = 'translateY(12px)';
  }, 1600);
}
'''

# 3. Replace alert in addBoxFromButton
old_add = "function addBoxFromButton() {\n  setTool('draw');\n  alert('💡 Bạn hãy kéo giữ chuột trái trên khung ảnh để vẽ hộp số mới!');\n}"
new_add = "function addBoxFromButton() {\n  setTool('draw');\n  showToast('Chế độ vẽ: Kéo giữ chuột trên ảnh để tạo box mới');\n}"

content = content.replace(old_add, new_add)

# 4. Replace alert in saveCurrent
old_save_alert = "    alert('✔️ Đã lưu điều chỉnh BBox và nhãn thành công!');"
new_save_alert = """    showToast('Đã lưu điều chỉnh thành công!');
    const saveBtn = document.querySelector('button[onclick="saveCurrent()"]');
    if (saveBtn) {
      const origText = saveBtn.innerHTML;
      saveBtn.innerHTML = '✔️ Đã lưu!';
      saveBtn.style.background = '#2ea043';
      setTimeout(() => {
        saveBtn.innerHTML = origText;
        saveBtn.style.background = '';
      }, 1000);
    }"""

content = content.replace(old_save_alert, new_save_alert)

# Add showToast definition before init()
content = content.replace('init();', toast_js + '\ninit();')

html_path.write_text(content, encoding="utf-8")
print("Successfully removed blocking alerts and added smooth toast notification!")

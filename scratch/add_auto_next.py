from pathlib import Path

html_path = Path("/home/deist/Downloads/OCR/anh/templates/validate1_edit.html")
content = html_path.read_text(encoding="utf-8")

# 1. Update button text
content = content.replace(
    '<button class="btn btn-primary" style="flex:1; padding:10px; font-size:13px;" onclick="saveCurrent()">💾 Lưu thay đổi (Ctrl+S)</button>',
    '<button class="btn btn-primary" style="flex:1; padding:10px; font-size:13px;" onclick="saveCurrent()">💾 Lưu & Sang ảnh tiếp (Ctrl+S)</button>'
)

# 2. Update saveCurrent to auto-advance
old_save_block = """  if (res.ok) {
    allEvals[name] = payload;
    updateCounts();
    renderSidebar();
    showToast('Đã lưu điều chỉnh thành công!');
    const saveBtn = document.querySelector('button[onclick="saveCurrent()"]');
    if (saveBtn) {
      const origText = saveBtn.innerHTML;
      saveBtn.innerHTML = '✔️ Đã lưu!';
      saveBtn.style.background = '#2ea043';
      setTimeout(() => {
        saveBtn.innerHTML = origText;
        saveBtn.style.background = '';
      }, 1000);
    }
  }"""

new_save_block = """  if (res.ok) {
    allEvals[name] = payload;
    updateCounts();
    renderSidebar();
    showToast('Đã lưu! Tự động chuyển tiếp ⏩');
    const saveBtn = document.querySelector('button[onclick="saveCurrent()"]');
    if (saveBtn) {
      const origText = saveBtn.innerHTML;
      saveBtn.innerHTML = '✔️ Đã lưu → Next!';
      saveBtn.style.background = '#2ea043';
      setTimeout(() => {
        saveBtn.innerHTML = origText;
        saveBtn.style.background = '';
      }, 800);
    }
    // Tự động chuyển sang ảnh tiếp theo sau 300ms
    setTimeout(() => {
      nextImage();
    }, 300);
  }"""

content = content.replace(old_save_block, new_save_block)

# 3. Add prevImage function
old_next_func = """function nextImage() {
  if (currentIdx < allImages.length - 1) {
    selectImage(currentIdx + 1);
  }
}"""

new_next_func = """function nextImage() {
  if (currentIdx < allImages.length - 1) {
    selectImage(currentIdx + 1);
  } else {
    showToast('🎉 Đã hoàn thành duyệt đến ảnh cuối cùng!');
  }
}

function prevImage() {
  if (currentIdx > 0) {
    selectImage(currentIdx - 1);
  }
}"""

content = content.replace(old_next_func, new_next_func)

# 4. Add Enter key support in inputs and Arrow keys
old_keydown = """document.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 's') {
    e.preventDefault();
    saveCurrent();
  } else if (e.key === 'Delete' || e.key === 'Backspace') {
    if (document.activeElement.tagName !== 'INPUT') {
      e.preventDefault();
      deleteSelectedBox();
    }
  } else if (e.key === 'v' || e.key === 'V') {
    if (document.activeElement.tagName !== 'INPUT') setTool('select');
  } else if (e.key === 'd' || e.key === 'D') {
    if (document.activeElement.tagName !== 'INPUT') setTool('draw');
  }
});"""

new_keydown = """document.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 's') {
    e.preventDefault();
    saveCurrent();
  } else if (e.key === 'Enter') {
    // Nhấn Enter ở bất kỳ ô input nào cũng sẽ tự động lưu và chuyển sang ảnh tiếp
    if (document.activeElement.tagName === 'INPUT') {
      e.preventDefault();
      saveCurrent();
    }
  } else if (e.key === 'Delete' || e.key === 'Backspace') {
    if (document.activeElement.tagName !== 'INPUT') {
      e.preventDefault();
      deleteSelectedBox();
    }
  } else if (e.key === 'v' || e.key === 'V') {
    if (document.activeElement.tagName !== 'INPUT') setTool('select');
  } else if (e.key === 'd' || e.key === 'D') {
    if (document.activeElement.tagName !== 'INPUT') setTool('draw');
  } else if (e.key === 'ArrowRight') {
    if (document.activeElement.tagName !== 'INPUT') {
      e.preventDefault();
      nextImage();
    }
  } else if (e.key === 'ArrowLeft') {
    if (document.activeElement.tagName !== 'INPUT') {
      e.preventDefault();
      prevImage();
    }
  }
});"""

content = content.replace(old_keydown, new_keydown)

html_path.write_text(content, encoding="utf-8")
print("Successfully added auto-advance and navigation shortcuts!")

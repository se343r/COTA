from pathlib import Path

html_path = Path("/home/deist/Downloads/OCR/anh/templates/testocr.html")
content = html_path.read_text(encoding="utf-8")

# 1. Update toolbar
old_toolbar = '''    <h3 style="margin:0;font-size:15px;">🔍 Test Pipeline (Full OCR)</h3>
    <div style="display:flex;gap:6px;align-items:center;">
      <div style="background:#1e252e; border:1px solid #3c495a; padding:2px 6px; border-radius:4px; display:flex; align-items:center; font-size:12px; height: 28px;">
        <span style="color:#8b98a8; margin-right:4px;">Elec Slices:</span>
        <input type="number" id="slice-count-input" value="6" min="4" max="9" style="width:32px; background:#0f1216; color:white; border:none; text-align:center;">
      </div>
      <button onclick="prevImage()">◀</button>
      <button onclick="nextImage()">▶</button>
      <button onclick="runInference()" style="background:#1a5a3a;">🚀 Detect</button>
      <button onclick="clearResults()" style="background:#5a1a1a;">🗑️ Xóa kết quả</button>
    </div>'''

new_toolbar = '''    <div style="display:flex; align-items:center; gap:10px;">
      <h3 style="margin:0;font-size:15px;">🔍 Test Pipeline (Full OCR)</h3>
      <select id="batch-select" onchange="changeBatch(this.value)" style="background:#161b22; color:#58a6ff; border:1px solid #30363d; padding:4px 8px; border-radius:4px; font-weight:bold; font-size:12px; cursor:pointer;">
        <option value="2">⚡ Validate 2 (50 ảnh mới)</option>
        <option value="1">⚙️ Validate 1 (48 ảnh cũ)</option>
      </select>
    </div>
    <div style="display:flex;gap:6px;align-items:center;">
      <a href="/validate1" style="background:#1f6feb; color:#fff; text-decoration:none; padding:5px 10px; border-radius:4px; font-size:12px; font-weight:600; display:flex; align-items:center; gap:4px; margin-right:6px;">🛠️ Sửa Validate 1</a>
      <div style="background:#1e252e; border:1px solid #3c495a; padding:2px 6px; border-radius:4px; display:flex; align-items:center; font-size:12px; height: 28px;">
        <span style="color:#8b98a8; margin-right:4px;">Elec Slices:</span>
        <input type="number" id="slice-count-input" value="6" min="4" max="9" style="width:32px; background:#0f1216; color:white; border:none; text-align:center;">
      </div>
      <button onclick="prevImage()">◀</button>
      <button onclick="nextImage()">▶</button>
      <button onclick="runInference()" style="background:#1a5a3a;">🚀 Detect</button>
      <button onclick="clearResults()" style="background:#5a1a1a;">🗑️ Xóa kết quả</button>
    </div>'''

content = content.replace(old_toolbar, new_toolbar)

# 2. Update JavaScript to support currentBatch (default '2')
old_js_head = '''let images = [];
let currentIdx = -1;
let currentImgObj = null;
let savedEvals = {};

async function loadList() {
  images = await fetch("/api/testocr/images").then(r => r.json());'''

new_js_head = '''let images = [];
let currentIdx = -1;
let currentImgObj = null;
let savedEvals = {};
const urlParams = new URLSearchParams(window.location.search);
let currentBatch = urlParams.get('batch') || "2";

function changeBatch(b) {
  currentBatch = b;
  const newUrl = new URL(window.location);
  newUrl.searchParams.set('batch', b);
  window.history.pushState({}, '', newUrl);
  loadList();
}

async function loadList() {
  const batchSel = document.getElementById("batch-select");
  if (batchSel) batchSel.value = currentBatch;
  
  images = await fetch(`/api/testocr/images?batch=${currentBatch}`).then(r => r.json());'''

content = content.replace(old_js_head, new_js_head)

# 3. Update all image/eval API calls to include batch parameter
content = content.replace(
    'img.src = "/api/testocr/image/" + name;',
    'img.src = `/api/testocr/image/${encodeURIComponent(name)}?batch=${currentBatch}`;'
)

content = content.replace(
    'const allEvals = await fetch("/api/testocr/all_evals").then(r => r.json());',
    'const allEvals = await fetch(`/api/testocr/all_evals?batch=${currentBatch}`).then(r => r.json());'
)

content = content.replace(
    'currentImgObj.src = "/api/testocr/image/" + name;',
    'currentImgObj.src = `/api/testocr/image/${encodeURIComponent(name)}?batch=${currentBatch}`;'
)

content = content.replace(
    'fetch("/api/testocr/infer/" + encodeURIComponent(name) + "?slice_count=" + sliceCount)',
    'fetch(`/api/testocr/infer/${encodeURIComponent(name)}?slice_count=${sliceCount}&batch=${currentBatch}`)'
)

content = content.replace(
    'fetch("/api/testocr/evaluate/" + encodeURIComponent(name), {',
    'fetch(`/api/testocr/evaluate/${encodeURIComponent(name)}?batch=${currentBatch}`, {'
)

html_path.write_text(content, encoding="utf-8")
print("testocr.html successfully patched!")

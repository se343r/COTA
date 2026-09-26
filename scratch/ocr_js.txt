/* ============================================================== */
/* OCR SCREEN                                                     */
/* ============================================================== */
let ocrImages = [];
let ocrIdx = 0;

let ocrImgData = null; // warped image
let ocrImgW = 0;
let ocrImgH = 0;
let ocrCorners = [];
let ocrMatrix = [];
let ocrDigits = []; // {cx, cy, w, h, label}

let ocrScale = 1;
let ocrSelIdx = -1;
let ocrDragging = null; // 'create', 'move', 'tl', 'tr', 'bl', 'br'
let ocrDragStart = null;
let ocrHover = null; // {idx, part}

function getCan() { return $("ocr-canvas"); }
function getCtx() { return $("ocr-canvas").getContext("2d"); }

async function bootOcrView() {
  const r = await fetch("/api/ocr_images").then(x => x.json());
  ocrImages = r.images;
  buildOcrList();
  updateOcrProgress();
  if (ocrImages.length) await ocrSelect(0);
}

function buildOcrList() {
  const list = $("ocr-list");
  list.innerHTML = "<h3>OCR images</h3>";
  ocrImages.forEach((item, i) => {
    const img = document.createElement("img");
    img.className = "tthumb";
    img.src = `/images/${encodeURIComponent(item.name)}/thumb`;
    if (item.text) img.classList.add("t-ok");
    img.addEventListener("click", () => ocrSelect(i));
    list.appendChild(img);
  });
}

function updateOcrProgress() {
  const done = ocrImages.filter(x => x.text).length;
  $("ocr-progress").textContent = `${done}/${ocrImages.length} ảnh đã gán chỉ số`;
}

async function ocrSelect(i) {
  if (i < 0 || i >= ocrImages.length) return;
  ocrIdx = i;
  const item = ocrImages[i];
  
  document.querySelectorAll("#ocr-list .tthumb").forEach((el, idx) => {
    el.classList.toggle("t-active", idx === i);
  });
  const active = $("ocr-list").querySelector(".tthumb.t-active");
  if (active) active.scrollIntoView({ block: "nearest" });

  $("ocr-meta-info").textContent = item.name;
  $("ocr-canvas").style.display = "none";
  ocrSelIdx = -1;
  ocrDigits = [];

  const r = await fetch(`/api/rectified_crop/${encodeURIComponent(item.name)}`).then(x => x.json());
  if (!r.ok) return;
  
  ocrCorners = r.corners;
  ocrMatrix = r.matrix;
  ocrImgW = r.w;
  ocrImgH = r.h;
  ocrDigits = r.digits || [];
  if (ocrDigits.length > 0) ocrSelIdx = 0;
  
  ocrImgData = new Image();
  ocrImgData.onload = () => {
    $("ocr-canvas").style.display = "";
    ocrFitCanvas();
    ocrDraw();
    updateOcrText();
  };
  ocrImgData.src = "data:image/jpeg;base64," + r.image_b64;
}

function ocrFitCanvas() {
  // Let's make the canvas height fixed at 200px or so
  const targetH = 200;
  ocrScale = targetH / ocrImgH;
  $("ocr-canvas").width = ocrImgW * ocrScale;
  $("ocr-canvas").height = targetH;
}

function getAABB(d) {
  // d has cx, cy, w, h normalized
  return {
    x: (d.cx - d.w/2) * ocrImgW * ocrScale,
    y: (d.cy - d.h/2) * ocrImgH * ocrScale,
    w: d.w * ocrImgW * ocrScale,
    h: d.h * ocrImgH * ocrScale
  };
}

function ocrDraw() {
  if (!ocrImgData) return;
  getCtx().clearRect(0, 0, $("ocr-canvas").width, $("ocr-canvas").height);
  getCtx().drawImage(ocrImgData, 0, 0, $("ocr-canvas").width, $("ocr-canvas").height);
  
  ocrDigits.forEach((d, i) => {
    const isSel = i === ocrSelIdx;
    const isHover = ocrHover && ocrHover.idx === i;
    const b = getAABB(d);
    
    getCtx().fillStyle = isSel ? "rgba(0,255,100,0.3)" : (isHover ? "rgba(255,255,255,0.2)" : "rgba(0,150,255,0.2)");
    getCtx().fillRect(b.x, b.y, b.w, b.h);
    
    getCtx().strokeStyle = isSel ? "#0f0" : "#0af";
    getCtx().lineWidth = isSel ? 2 : 1;
    getCtx().strokeRect(b.x, b.y, b.w, b.h);
    
    // Draw label
    if (d.label) {
      getCtx().fillStyle = isSel ? "#0f0" : "#fff";
      getCtx().font = "bold 24px monospace";
      getCtx().textAlign = "center";
      getCtx().textBaseline = "middle";
      getCtx().fillText(d.label.toUpperCase(), b.x + b.w/2, b.y + b.h/2);
    }
    
    // Handles if selected
    if (isSel) {
      const hs = 8;
      getCtx().fillStyle = "#fff";
      const pts = [
        {x: b.x, y: b.y}, // tl
        {x: b.x+b.w, y: b.y}, // tr
        {x: b.x, y: b.y+b.h}, // bl
        {x: b.x+b.w, y: b.y+b.h} // br
      ];
      pts.forEach(p => getCtx().fillRect(p.x - hs/2, p.y - hs/2, hs, hs));
    }
  });
}

function updateOcrText() {
  const txt = ocrDigits.map(d => d.label ? (d.label === "unreadable" ? "_" : d.label) : "?").join("");
  $("ocr-result-text").textContent = txt || "---";
}

// Interaction
function ocrCanPt(e) {
  const r = $("ocr-canvas").getBoundingClientRect();
  return { x: e.clientX - r.left, y: e.clientY - r.top };
}

function ocrHitTest(pt) {
  // Check handles of selected
  if (ocrSelIdx >= 0 && ocrSelIdx < ocrDigits.length) {
    const b = getAABB(ocrDigits[ocrSelIdx]);
    const hs = 8;
    const check = (x,y) => Math.abs(pt.x - x) <= hs && Math.abs(pt.y - y) <= hs;
    if (check(b.x, b.y)) return {idx: ocrSelIdx, part: "tl"};
    if (check(b.x+b.w, b.y)) return {idx: ocrSelIdx, part: "tr"};
    if (check(b.x, b.y+b.h)) return {idx: ocrSelIdx, part: "bl"};
    if (check(b.x+b.w, b.y+b.h)) return {idx: ocrSelIdx, part: "br"};
  }
  
  // Check bodies
  for (let i = ocrDigits.length - 1; i >= 0; i--) {
    const b = getAABB(ocrDigits[i]);
    if (pt.x >= b.x && pt.x <= b.x+b.w && pt.y >= b.y && pt.y <= b.y+b.h) {
      return {idx: i, part: "body"};
    }
  }
  return null;
}

$("ocr-canvas").addEventListener("mousemove", e => {
  const pt = ocrCanPt(e);
  ocrHover = ocrHitTest(pt);
  
  if (ocrDragging === 'create' && ocrDragStart) {
    $("ocr-canvas").style.cursor = 'crosshair';
    // Update the newly created box (which is the last one)
    const b = ocrDigits[ocrDigits.length - 1];
    const x0 = Math.min(pt.x, ocrDragStart.x) / (ocrImgW * ocrScale);
    const y0 = Math.min(pt.y, ocrDragStart.y) / (ocrImgH * ocrScale);
    const x1 = Math.max(pt.x, ocrDragStart.x) / (ocrImgW * ocrScale);
    const y1 = Math.max(pt.y, ocrDragStart.y) / (ocrImgH * ocrScale);
    b.cx = (x0 + x1) / 2;
    b.cy = (y0 + y1) / 2;
    b.w = x1 - x0;
    b.h = y1 - y0;
  } else if (ocrDragging && ocrSelIdx >= 0) {
    const b = ocrDigits[ocrSelIdx];
    let x0 = (b.cx - b.w/2);
    let y0 = (b.cy - b.h/2);
    let x1 = (b.cx + b.w/2);
    let y1 = (b.cy + b.h/2);
    
    const nx = pt.x / (ocrImgW * ocrScale);
    const ny = pt.y / (ocrImgH * ocrScale);
    
    if (ocrDragging === 'move') {
      const dx = (pt.x - ocrDragStart.x) / (ocrImgW * ocrScale);
      const dy = (pt.y - ocrDragStart.y) / (ocrImgH * ocrScale);
      b.cx += dx; b.cy += dy;
      ocrDragStart = pt;
    } else {
      if (ocrDragging === 'tl') { x0 = nx; y0 = ny; }
      if (ocrDragging === 'tr') { x1 = nx; y0 = ny; }
      if (ocrDragging === 'bl') { x0 = nx; y1 = ny; }
      if (ocrDragging === 'br') { x1 = nx; y1 = ny; }
      
      const fx0 = Math.min(x0, x1), fx1 = Math.max(x0, x1);
      const fy0 = Math.min(y0, y1), fy1 = Math.max(y0, y1);
      b.cx = (fx0 + fx1)/2; b.cy = (fy0 + fy1)/2;
      b.w = fx1 - fx0; b.h = fy1 - fy0;
    }
  } else {
    // Update cursor based on hover
    if (ocrHover) {
      if (ocrHover.part.startsWith("t") || ocrHover.part.startsWith("b")) $("ocr-canvas").style.cursor = "pointer";
      else $("ocr-canvas").style.cursor = "move";
    } else {
      $("ocr-canvas").style.cursor = "crosshair";
    }
  }
  ocrDraw();
});

$("ocr-canvas").addEventListener("mousedown", e => {
  const pt = ocrCanPt(e);
  ocrDragStart = pt;
  const hit = ocrHitTest(pt);
  
  if (hit) {
    ocrSelIdx = hit.idx;
    ocrDragging = hit.part === "body" ? "move" : hit.part;
  } else {
    // Start creating a new box
    ocrDigits.push({ cx: pt.x / (ocrImgW * ocrScale), cy: pt.y / (ocrImgH * ocrScale), w: 0.01, h: 0.01, label: "" });
    ocrSelIdx = ocrDigits.length - 1;
    ocrDragging = 'create';
  }
  ocrDraw();
  $("ocr-main").focus();
});

window.addEventListener("mouseup", e => {
  if (ocrDragging) {
    ocrDragging = null;
    ocrDragStart = null;
    
    // Sort boxes left to right
    const selBox = ocrDigits[ocrSelIdx];
    ocrDigits.sort((a,b) => a.cx - b.cx);
    ocrSelIdx = ocrDigits.indexOf(selBox);
    
    updateOcrText();
    ocrDraw();
  }
});

$("ocr-main").addEventListener("keydown", e => {
  const inOcr = $("ocr-view").classList.contains("active");
  if (!inOcr) return;
  
  // Handle shortcuts
  if (e.key === "Enter") { e.preventDefault(); ocrSave(); return; }
  
  if (ocrSelIdx >= 0 && ocrSelIdx < ocrDigits.length) {
    const k = e.key.toLowerCase();
    if (k >= "0" && k <= "9") {
      ocrDigits[ocrSelIdx].label = k;
      if (ocrSelIdx < ocrDigits.length - 1) ocrSelIdx++;
      updateOcrText();
      ocrDraw();
    } else if (k === "u") {
      ocrDigits[ocrSelIdx].label = "unreadable";
      if (ocrSelIdx < ocrDigits.length - 1) ocrSelIdx++;
      updateOcrText();
      ocrDraw();
    } else if (e.key === "Delete" || e.key === "Backspace") {
      ocrDigits.splice(ocrSelIdx, 1);
      if (ocrSelIdx >= ocrDigits.length) ocrSelIdx = ocrDigits.length - 1;
      updateOcrText();
      ocrDraw();
    } else if (e.key === "ArrowLeft") {
      ocrSelIdx = Math.max(0, ocrSelIdx - 1);
      ocrDraw();
    } else if (e.key === "ArrowRight") {
      ocrSelIdx = Math.min(ocrDigits.length - 1, ocrSelIdx + 1);
      ocrDraw();
    }
  }
});

async function ocrSave() {
  if (!ocrImages.length) return;
  const item = ocrImages[ocrIdx];
  
  await fetch("/api/digit_labels", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ 
      name: item.name, 
      corners: ocrCorners, 
      matrix: ocrMatrix, 
      digits: ocrDigits 
    })
  });
  
  const text = ocrDigits.filter(d => d.label !== "unreadable").map(d => d.label).join("");
  item.text = text;
  const thumb = $("ocr-list").querySelectorAll(".tthumb")[ocrIdx];
  if (thumb) {
    if (text) thumb.classList.add("t-ok");
    else thumb.classList.remove("t-ok");
  }
  updateOcrProgress();
  ocrSelect(ocrIdx + 1);
}

$("btn-ocr-save").addEventListener("click", ocrSave);
$("btn-ocr-skip").addEventListener("click", () => ocrSelect(ocrIdx + 1));

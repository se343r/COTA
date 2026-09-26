let ocrCreateStart = null;

function ocrCanPt(e) {
  const r = $("ocr-canvas").getBoundingClientRect();
  return { x: e.clientX - r.left, y: e.clientY - r.top };
}

function ocrHitTest(pt) {
  if (ocrSelIdx >= 0 && ocrSelIdx < ocrDigits.length) {
    const b = getAABB(ocrDigits[ocrSelIdx]);
    const hs = 8;
    const check = (x,y) => Math.abs(pt.x - x) <= hs && Math.abs(pt.y - y) <= hs;
    if (check(b.x, b.y)) return {idx: ocrSelIdx, part: "tl"};
    if (check(b.x+b.w, b.y)) return {idx: ocrSelIdx, part: "tr"};
    if (check(b.x, b.y+b.h)) return {idx: ocrSelIdx, part: "bl"};
    if (check(b.x+b.w, b.y+b.h)) return {idx: ocrSelIdx, part: "br"};
  }
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
  
  if (ocrDragging === 'create' || ocrCreateStart) {
    $("ocr-canvas").style.cursor = 'crosshair';
    const start = ocrCreateStart || ocrDragStart;
    const b = ocrDigits[ocrDigits.length - 1];
    const x0 = Math.min(pt.x, start.x) / (ocrImgW * ocrScale);
    const y0 = Math.min(pt.y, start.y) / (ocrImgH * ocrScale);
    const x1 = Math.max(pt.x, start.x) / (ocrImgW * ocrScale);
    const y1 = Math.max(pt.y, start.y) / (ocrImgH * ocrScale);
    b.cx = (x0 + x1) / 2; b.cy = (y0 + y1) / 2;
    b.w = x1 - x0; b.h = y1 - y0;
  } else if (ocrDragging && ocrSelIdx >= 0) {
    const b = ocrDigits[ocrSelIdx];
    let x0 = (b.cx - b.w/2); let y0 = (b.cy - b.h/2);
    let x1 = (b.cx + b.w/2); let y1 = (b.cy + b.h/2);
    
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
  
  if (ocrCreateStart) {
    ocrCreateStart = null;
    ocrDragging = null;
    if (ocrSelIdx >= 0 && ocrSelIdx < ocrDigits.length) {
      const selBox = ocrDigits[ocrSelIdx];
      ocrDigits.sort((a,b) => a.cx - b.cx);
      ocrSelIdx = ocrDigits.indexOf(selBox);
    }
    updateOcrText();
    ocrDraw();
    $("ocr-main").focus();
    return;
  }
  
  ocrDragStart = pt;
  const hit = ocrHitTest(pt);
  if (hit) {
    ocrSelIdx = hit.idx;
    ocrDragging = hit.part === "body" ? "move" : hit.part;
  } else {
    ocrDigits.push({ cx: pt.x / (ocrImgW * ocrScale), cy: pt.y / (ocrImgH * ocrScale), w: 0, h: 0, label: "" });
    ocrSelIdx = ocrDigits.length - 1;
    ocrDragging = 'create';
  }
  ocrDraw();
  $("ocr-main").focus();
});

window.addEventListener("mouseup", e => {
  if (ocrDragging === 'create') {
    const b = ocrDigits[ocrDigits.length - 1];
    const pixelW = b.w * ocrImgW * ocrScale;
    const pixelH = b.h * ocrImgH * ocrScale;
    if (pixelW < 5 || pixelH < 5) {
      ocrCreateStart = ocrDragStart;
      ocrDragging = null;
      return;
    }
  }
  
  if (ocrDragging || ocrCreateStart) {
    if (ocrCreateStart && e.target.id !== "ocr-canvas") {
      ocrCreateStart = null; // Cancel if clicked outside
    }
    ocrDragging = null;
    ocrDragStart = null;
    if (ocrSelIdx >= 0 && ocrSelIdx < ocrDigits.length) {
      const selBox = ocrDigits[ocrSelIdx];
      ocrDigits.sort((a,b) => a.cx - b.cx);
      ocrSelIdx = ocrDigits.indexOf(selBox);
    }
    updateOcrText();
    ocrDraw();
  }
});

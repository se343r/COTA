$("ocr-canvas").addEventListener("mousemove", e => {
  if (ocrClassId === 1) return;
  const pt = ocrCanPt(e);
  // ... rest

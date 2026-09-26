$("ocr-main").addEventListener("keydown", e => {
  const inOcr = $("ocr-view").classList.contains("active");
  if (!inOcr) return;
  
  if (e.key === "Enter") { e.preventDefault(); ocrSave(); return; }
  
  if (ocrSelIdx >= 0 && ocrSelIdx < ocrDigits.length) {
    const k = e.key.toLowerCase();
    const d = ocrDigits[ocrSelIdx];
    
    if (k >= "0" && k <= "9") {
      d.label = k;
      if (ocrSelIdx < ocrDigits.length - 1) ocrSelIdx++;
      updateOcrText(); ocrDraw();
    } else if (k === "u") {
      d.label = "unreadable";
      if (ocrSelIdx < ocrDigits.length - 1) ocrSelIdx++;
      updateOcrText(); ocrDraw();
    } else if (k === "b") {
      d.is_between = !d.is_between;
      updateOcrText(); ocrDraw();
    } else if (e.key === "." || e.key === ",") {
      d.is_decimal = !d.is_decimal;
      updateOcrText(); ocrDraw();
    } else if (e.key === "Delete" || e.key === "Backspace") {
      ocrDigits.splice(ocrSelIdx, 1);
      if (ocrSelIdx >= ocrDigits.length) ocrSelIdx = ocrDigits.length - 1;
      updateOcrText(); ocrDraw();
    } else if (e.key === "ArrowLeft") {
      ocrSelIdx = Math.max(0, ocrSelIdx - 1);
      ocrDraw();
    } else if (e.key === "ArrowRight") {
      ocrSelIdx = Math.min(ocrDigits.length - 1, ocrSelIdx + 1);
      ocrDraw();
    }
  }
});

let ocrClassId = 0;

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
  ocrClassId = r.class_id || 0;
  ocrDigits = r.digits || [];
  
  if (ocrClassId === 1) { // Điện tử
    $("ocr-mode-badge").textContent = "Điện tử (Chuỗi)";
    $("ocr-mode-badge").style.background = "#0055aa";
    $("ocr-mech-hint").style.display = "none";
    $("ocr-result-text").style.display = "none";
    $("ocr-string-input").style.display = "";
    $("ocr-string-input").value = r.full_text || "";
    $("ocr-string-input").focus();
  } else { // Cơ
    $("ocr-mode-badge").textContent = "Cơ (Từng số)";
    $("ocr-mode-badge").style.background = "#aa5500";
    $("ocr-mech-hint").style.display = "";
    $("ocr-result-text").style.display = "";
    $("ocr-string-input").style.display = "none";
    if (ocrDigits.length > 0) ocrSelIdx = 0;
    $("ocr-main").focus();
  }
  
  ocrImgData = new Image();
  ocrImgData.onload = () => {
    $("ocr-canvas").style.display = "";
    ocrFitCanvas();
    ocrDraw();
    if (ocrClassId !== 1) updateOcrText();
  };
  ocrImgData.src = "data:image/jpeg;base64," + r.image_b64;
}

async function ocrSave() {
  if (!ocrImages.length) return;
  const item = ocrImages[ocrIdx];
  const fullText = $("ocr-string-input").value.trim();
  
  await fetch("/api/digit_labels", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ 
      name: item.name, 
      corners: ocrCorners, 
      matrix: ocrMatrix, 
      class_id: ocrClassId,
      digits: ocrDigits,
      full_text: fullText
    })
  });
  
  let text = "";
  if (ocrClassId === 1) {
    text = fullText;
  } else {
    text = ocrDigits.map(d => {
      let t = d.label ? (d.label === "unreadable" ? "_" : d.label) : "?";
      if (d.is_between) t += "↕";
      if (d.is_decimal) t = "." + t;
      return t;
    }).join("");
  }
  
  item.text = text;
  const thumb = $("ocr-list").querySelectorAll(".tthumb")[ocrIdx];
  if (thumb) {
    if (text) thumb.classList.add("t-ok");
    else thumb.classList.remove("t-ok");
  }
  updateOcrProgress();
  ocrSelect(ocrIdx + 1);
}

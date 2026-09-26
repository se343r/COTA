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

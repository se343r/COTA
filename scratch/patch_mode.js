  if (ocrClassId === 1) { // Điện tử
    $("ocr-mode-badge").textContent = "Điện tử (Chuỗi)";
    $("ocr-mode-badge").style.background = "#0055aa";
    $("ocr-mech-hint").style.display = "";
    $("ocr-mech-hint").innerHTML = "Gõ dãy số vào ô bên dưới.<br><i>(Tùy chọn)</i> Nếu ảnh có nhiều chữ rác, hãy kéo chuột vẽ 1 box bao trọn dãy số.";
    $("ocr-result-text").style.display = "none";
    $("ocr-string-input").style.display = "";
    $("ocr-string-input").value = r.full_text || "";
    $("ocr-string-input").focus();
  } else { // Cơ
    $("ocr-mode-badge").textContent = "Cơ (Từng số)";
    $("ocr-mode-badge").style.background = "#aa5500";
    $("ocr-mech-hint").style.display = "";
    $("ocr-mech-hint").innerHTML = "Kéo chuột vẽ box. Click để chọn. Nhấn <code>0-9</code> gán số, <code>U</code> cho unreadable.<br>Nhấn <code>B</code> nếu số bị cuộn giữa 2 giá trị. Nhấn <code>.</code> nếu là số thập phân.<br>Nhấn <code>Delete</code> để xóa box. Kéo các góc để thu phóng.";
    $("ocr-result-text").style.display = "";
    $("ocr-string-input").style.display = "none";
    if (ocrDigits.length > 0) ocrSelIdx = 0;
    $("ocr-main").focus();
  }

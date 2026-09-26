import json, math

with open("/home/deist/Downloads/OCR/anh/scratch/full_stats.json") as f:
    stats = json.load(f)

# Total images
total_images = stats["total_images"]

# OBB and BBOX across ALL types
total_obb_ok = stats["obb_ok"]
total_bbox_ok = stats["bbox_ok"]
obb_pct = total_obb_ok / total_images * 100 if total_images else 0
bbox_pct = total_bbox_ok / total_images * 100 if total_images else 0

# OCR calculation (combining mechanical digits + electronic chars)
# Mechanical
mech_evaluated = stats["mech"]["ocr_evaluated"]
mech_correct = stats["mech"]["ocr_correct"]

# Electronic
elec_evaluated = stats["elec"]["ocr_evaluated_chars"]
elec_correct = stats["elec"]["ocr_correct_chars"]

total_ocr_evaluated = mech_evaluated + elec_evaluated
total_ocr_correct = mech_correct + elec_correct
ocr_pct = total_ocr_correct / total_ocr_evaluated * 100 if total_ocr_evaluated else 0

# End-to-end
total_e2e_ok = stats["mech"]["e2e_ok"] + stats["elec"]["e2e_ok"]
# Wait, e2e percentage in the funnel is traditionally calculated as the product of the stages, OR as the absolute pass rate.
# The user's original HTML had ~63% and = 0.958 x 0.667 x 0.965.
# Let's keep it as the product to match their style.
e2e_pct = (obb_pct/100) * (bbox_pct/100) * (ocr_pct/100) * 100

# BBox failure reasons (combined)
missing_total = total_images - total_bbox_ok
err_half = stats["mech"]["errors"]["half"]
err_blur = stats["mech"]["errors"]["blurry"] + stats["elec"]["errors"]["blurry"]
err_out = stats["mech"]["errors"]["out"] + stats["elec"]["errors"]["out"]
err_other = missing_total - (err_half + err_blur + err_out)

p_half = err_half / missing_total * 100 if missing_total else 0
p_blur = err_blur / missing_total * 100 if missing_total else 0
p_out = err_out / missing_total * 100 if missing_total else 0
p_other = err_other / missing_total * 100 if missing_total else 0

# Digits chart
mech_clean = stats["mech"]["digits_clean"]
mech_half = stats["mech"]["digits_half"]
mech_sp = stats["mech"]["digits_sp"]
mech_total_digits = stats["mech"]["digits_total"]
# OCR Error for the chart = total evaluated digits - correct digits
# wait, OCR Error = mech_evaluated - mech_correct?
ocr_errors = mech_evaluated - mech_correct

pie_data = f"""[
    {{ value: {stats["types"].get("mechanical", 0)}, color: "#3b82f6" }},
    {{ value: {stats["types"].get("electronic", 0)}, color: "#f59e0b" }},
    {{ value: {total_images - stats["types"].get("mechanical", 0) - stats["types"].get("electronic", 0)}, color: "#6b7280" }},
]"""

bars_data = f"""[
    {{ label: "Clean\\nDigits", value: {mech_clean}, total: {mech_total_digits}, color: "#22c55e" }},
    {{ label: "Half\\nDigit", value: {mech_half}, total: {mech_total_digits}, color: "#f59e0b" }},
    {{ label: "Spurious\\n(FP)", value: {mech_sp}, total: {mech_total_digits}, color: "#ef4444" }},
    {{ label: "OCR\\nError", value: {ocr_errors}, total: {mech_total_digits}, color: "#8b5cf6" }},
]"""

# Per img list (include both mechanical and electronic that passed bbox)
js_array = "[\n"
for r in stats["per_img"]:
    # color electronic differently or just show them together
    js_array += f'  {{img:"{r["img"]}",actual:"{r["actual"]}",pred:"{r["pred"]}",type:"{r["type"]}",n:{r["n"]},half:{r["half"]},sp:{r["sp"]}}},\n'
js_array += "]"

html = f"""<!DOCTYPE html>
<html>
<head>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
</head>
<body class="bg-[var(--background)] text-[var(--foreground)] antialiased p-6">

<div class="max-w-5xl mx-auto space-y-6">

  <!-- Header -->
  <div>
    <h1 class="text-2xl font-bold">📊 Báo cáo Đánh giá Pipeline OCR (Bao gồm Điện tử)</h1>
    <p class="text-[var(--muted-foreground)] text-sm mt-1">Dựa trên <strong>{total_images} ảnh</strong> được đánh giá thủ công</p>
  </div>

  <!-- Row 1: KPI cards -->
  <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
    <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-4 text-center">
      <div class="text-3xl font-bold text-[var(--primary)]">{total_images}</div>
      <div class="text-xs text-[var(--muted-foreground)] mt-1">Tổng ảnh</div>
    </div>
    <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-4 text-center">
      <div class="text-3xl font-bold" style="color:#22c55e">{obb_pct:.1f}%</div>
      <div class="text-xs text-[var(--muted-foreground)] mt-1">OBB Accuracy</div>
      <div class="text-xs text-[var(--muted-foreground)]">{total_obb_ok}/{total_images}</div>
    </div>
    <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-4 text-center">
      <div class="text-3xl font-bold" style="color:#f59e0b">{bbox_pct:.1f}%</div>
      <div class="text-xs text-[var(--muted-foreground)] mt-1">BBox Accuracy</div>
      <div class="text-xs text-[var(--muted-foreground)]">{total_bbox_ok}/{total_images}</div>
    </div>
    <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-4 text-center">
      <div class="text-3xl font-bold" style="color:#22c55e">{ocr_pct:.1f}%</div>
      <div class="text-xs text-[var(--muted-foreground)] mt-1">OCR Accuracy (Kí tự)</div>
      <div class="text-xs text-[var(--muted-foreground)]">{total_ocr_correct}/{total_ocr_evaluated} ký tự</div>
    </div>
  </div>

  <!-- Row 2: Meter type + BBox failure -->
  <div class="grid grid-cols-1 md:grid-cols-2 gap-4">

    <!-- Meter type pie -->
    <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-5">
      <h2 class="font-semibold mb-4">🏷️ Phân loại ảnh</h2>
      <div class="flex items-center gap-6">
        <canvas id="typePie" width="140" height="140"></canvas>
        <div class="space-y-2 text-sm">
          <div class="flex items-center gap-2"><span class="w-3 h-3 rounded-full inline-block" style="background:#3b82f6"></span> Công tơ cơ: <strong>{stats['types'].get('mechanical',0)}</strong></div>
          <div class="flex items-center gap-2"><span class="w-3 h-3 rounded-full inline-block" style="background:#f59e0b"></span> Công tơ điện tử: <strong>{stats['types'].get('electronic',0)}</strong></div>
          <div class="flex items-center gap-2"><span class="w-3 h-3 rounded-full inline-block" style="background:#6b7280"></span> Rác/Khác: <strong>{total_images - stats['types'].get('mechanical',0) - stats['types'].get('electronic',0)}</strong></div>
        </div>
      </div>
    </div>

    <!-- BBox failure reasons -->
    <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-5">
      <h2 class="font-semibold mb-4">⚠️ Lý do BBox detect thiếu <span class="text-[var(--muted-foreground)] font-normal text-xs">({missing_total} ca thất bại)</span></h2>
      <div class="space-y-3">
        <div>
          <div class="flex justify-between text-sm mb-1"><span>🔄 Half-digit chưa train (cơ)</span><span class="font-bold">{err_half}</span></div>
          <div class="w-full bg-[var(--border)] rounded-full h-2"><div class="h-2 rounded-full" style="width:{p_half}%;background:#ef4444"></div></div>
        </div>
        <div>
          <div class="flex justify-between text-sm mb-1"><span>🌫️ Bị mờ / bị che</span><span class="font-bold">{err_blur}</span></div>
          <div class="w-full bg-[var(--border)] rounded-full h-2"><div class="h-2 rounded-full" style="width:{p_blur}%;background:#f59e0b"></div></div>
        </div>
        <div>
          <div class="flex justify-between text-sm mb-1"><span>✂️ Ngoài khung OBB</span><span class="font-bold">{err_out}</span></div>
          <div class="w-full bg-[var(--border)] rounded-full h-2"><div class="h-2 rounded-full" style="width:{p_out}%;background:#8b5cf6"></div></div>
        </div>
        <div>
          <div class="flex justify-between text-sm mb-1"><span>❓ Khác</span><span class="font-bold">{err_other}</span></div>
          <div class="w-full bg-[var(--border)] rounded-full h-2"><div class="h-2 rounded-full" style="width:{p_other}%;background:#6b7280"></div></div>
        </div>
      </div>
    </div>
  </div>

  <!-- Row 3: Pipeline funnel + Digit anomalies -->
  <div class="grid grid-cols-1 md:grid-cols-2 gap-4">

    <!-- Pipeline funnel -->
    <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-5">
      <h2 class="font-semibold mb-4">🔗 Pipeline Accuracy Funnel (Tất cả)</h2>
      <div class="space-y-3">
        <div>
          <div class="flex justify-between text-sm mb-1">
            <span>1️⃣ OBB tìm màn hình</span>
            <span class="font-bold" style="color:#22c55e">{obb_pct:.1f}% ({total_obb_ok}/{total_images})</span>
          </div>
          <div class="w-full bg-[var(--border)] rounded-full h-4">
            <div class="h-4 rounded-full flex items-center justify-end pr-2 text-xs text-white font-bold" style="width:{obb_pct}%;background:linear-gradient(90deg,#16a34a,#22c55e)">{obb_pct:.1f}%</div>
          </div>
        </div>
        <div>
          <div class="flex justify-between text-sm mb-1">
            <span>2️⃣ BBox detect đúng vùng OCR</span>
            <span class="font-bold" style="color:#f59e0b">{bbox_pct:.1f}% ({total_bbox_ok}/{total_images})</span>
          </div>
          <div class="w-full bg-[var(--border)] rounded-full h-4">
            <div class="h-4 rounded-full flex items-center justify-end pr-2 text-xs text-white font-bold" style="width:{bbox_pct}%;background:linear-gradient(90deg,#d97706,#f59e0b)">{bbox_pct:.1f}%</div>
          </div>
        </div>
        <div>
          <div class="flex justify-between text-sm mb-1">
            <span>3️⃣ OCR đọc đúng ký tự</span>
            <span class="font-bold" style="color:#22c55e">{ocr_pct:.1f}% ({total_ocr_correct}/{total_ocr_evaluated})</span>
          </div>
          <div class="w-full bg-[var(--border)] rounded-full h-4">
            <div class="h-4 rounded-full flex items-center justify-end pr-2 text-xs text-white font-bold" style="width:{ocr_pct}%;background:linear-gradient(90deg,#16a34a,#22c55e)">{ocr_pct:.1f}%</div>
          </div>
        </div>
        <div class="pt-2 border-t border-[var(--border)]">
          <div class="flex justify-between text-sm mb-1">
            <span class="font-semibold">🎯 End-to-end (đủ pipeline)</span>
            <span class="font-bold" style="color:#ef4444">~{e2e_pct:.0f}%</span>
          </div>
          <div class="w-full bg-[var(--border)] rounded-full h-4">
            <div class="h-4 rounded-full flex items-center justify-end pr-2 text-xs text-white font-bold" style="width:{e2e_pct}%;background:linear-gradient(90deg,#dc2626,#ef4444)">~{e2e_pct:.0f}%</div>
          </div>
          <p class="text-xs text-[var(--muted-foreground)] mt-1">= {obb_pct/100:.3f} × {bbox_pct/100:.3f} × {ocr_pct/100:.3f}</p>
        </div>
      </div>
    </div>

    <!-- Digit anomalies breakdown -->
    <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-5">
      <h2 class="font-semibold mb-4">🔢 Phân tích {mech_total_digits} digit (riêng công tơ cơ)</h2>
      <canvas id="digitBar" width="340" height="180"></canvas>
    </div>
  </div>

  <!-- Row 4: Per-image table -->
  <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-5">
    <h2 class="font-semibold mb-3">📋 Kết quả từng ảnh (Cơ & Điện tử đã qua BBox)</h2>
    <div class="overflow-x-auto">
      <table class="w-full text-sm">
        <thead>
          <tr class="text-left text-[var(--muted-foreground)] border-b border-[var(--border)]">
            <th class="pb-2 pr-3">#</th>
            <th class="pb-2 pr-3">Ảnh</th>
            <th class="pb-2 pr-3">Thực tế</th>
            <th class="pb-2 pr-3">Mô hình đoán</th>
            <th class="pb-2 pr-3">Loại</th>
            <th class="pb-2 pr-3">Ký tự</th>
            <th class="pb-2 pr-3">Lỗi</th>
          </tr>
        </thead>
        <tbody id="imgTable" class="divide-y divide-[var(--border)]"></tbody>
      </table>
    </div>
  </div>

</div>

<script>
// ---- Data ----
const perImg = {js_array};

// Fill table
const tbody = document.getElementById("imgTable");
perImg.forEach((row, i) => {{
  const isOk = row.actual === row.pred;
  const isElec = row.type === "elec";
  const tr = document.createElement("tr");
  tr.innerHTML = `
    <td class="py-2 pr-3 text-[var(--muted-foreground)]">${{i+1}}</td>
    <td class="py-2 pr-3 font-mono text-xs">${{row.img}}</td>
    <td class="py-2 pr-3 font-mono font-bold text-lg tracking-widest" style="color:#4ade80">${{row.actual}}</td>
    <td class="py-2 pr-3 font-mono font-bold text-lg tracking-widest" style="color:${{isOk ? '#4ade80' : '#f87171'}}">${{row.pred}}</td>
    <td class="py-2 pr-3">${{isElec ? '<span class="text-xs px-2 py-1 rounded bg-[#f59e0b] text-[#451a03] font-bold">Điện tử</span>' : '<span class="text-xs px-2 py-1 rounded bg-[#3b82f6] text-[#0f172a] font-bold">Cơ</span>'}}</td>
    <td class="py-2 pr-3"><span class="px-2 py-0.5 rounded text-xs font-bold" style="background:#1e3a5f;color:#60a5fa">${{row.n}}</span></td>
    <td class="py-2 pr-3">
      ${{row.half > 0 ? `<span class="px-1 py-0.5 rounded text-xs font-bold" style="background:#422006;color:#fb923c">⚠️${{row.half}}</span>` : ''}}
      ${{row.sp > 0 ? `<span class="px-1 py-0.5 rounded text-xs font-bold" style="background:#450a0a;color:#f87171">❌${{row.sp}}</span>` : ''}}
      ${{row.half===0 && row.sp===0 ? '<span class="text-[var(--muted-foreground)]">—</span>' : ''}}
    </td>
  `;
  tbody.appendChild(tr);
}});

// ---- Pie chart: meter types ----
(function() {{
  const canvas = document.getElementById("typePie");
  if(!canvas) return;
  const ctx = canvas.getContext("2d");
  const isLight = document.documentElement.classList.contains('light');
  const data = {pie_data};
  const total = data.reduce((s, d) => s + d.value, 0);
  let startAngle = -Math.PI / 2;
  const cx = 70, cy = 70, r = 60;
  data.forEach(d => {{
    const sweep = (d.value / total) * 2 * Math.PI;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, r, startAngle, startAngle + sweep);
    ctx.closePath();
    ctx.fillStyle = d.color;
    ctx.fill();
    startAngle += sweep;
  }});
  // donut hole
  ctx.beginPath();
  ctx.arc(cx, cy, r * 0.55, 0, 2 * Math.PI);
  ctx.fillStyle = isLight ? "#fff" : "#1a1f2e";
  ctx.fill();
  ctx.fillStyle = isLight ? "#111" : "#fff";
  ctx.font = "bold 18px sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(total, cx, cy - 6);
  ctx.font = "11px sans-serif";
  ctx.fillStyle = "#9ca3af";
  ctx.fillText("ảnh", cx, cy + 10);
}})();

// ---- Bar chart: digit anomalies ----
(function() {{
  const canvas = document.getElementById("digitBar");
  if(!canvas) return;
  const ctx = canvas.getContext("2d");
  const isLight = document.documentElement.classList.contains('light');
  const barColor = isLight ? "#111" : "#fff";
  const mutedColor = "#9ca3af";
  const W = canvas.width, H = canvas.height;
  const pad = {{ top: 20, bottom: 40, left: 10, right: 10 }};
  const chartH = H - pad.top - pad.bottom;
  const chartW = W - pad.left - pad.right;

  const bars = {bars_data};
  const maxVal = Math.max(...bars.map(b => b.total), 1);
  const bw = chartW / bars.length;

  bars.forEach((b, i) => {{
    const x = pad.left + i * bw + bw * 0.15;
    const bWidth = bw * 0.7;
    const bH = (b.value / maxVal) * chartH;
    const y = pad.top + chartH - bH;

    // bar
    ctx.fillStyle = b.color;
    ctx.beginPath();
    ctx.roundRect(x, y, bWidth, bH, [4,4,0,0]);
    ctx.fill();

    // value label on top
    ctx.fillStyle = isLight ? "#111" : "#e5e7eb";
    ctx.font = "bold 13px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(b.value, x + bWidth/2, y - 5);

    // x-axis label
    ctx.fillStyle = mutedColor;
    ctx.font = "11px sans-serif";
    const lines = b.label.split("\\n");
    lines.forEach((line, li) => {{
      ctx.fillText(line, x + bWidth/2, pad.top + chartH + 14 + li * 13);
    }});
  }});
}})();
</script>

</body>
</html>
"""

with open("/home/deist/.gemini/antigravity/brain/c327d66a-ae54-4b63-a997-fec47b910363/eval_dashboard_v4.html", "w") as f:
    f.write(html)


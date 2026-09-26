import json, re

with open("/home/deist/Downloads/OCR/anh/testocr_eval.json") as f:
    eval_data = json.load(f)

total_images = len(eval_data)
types = {"mechanical": 0, "electronic": 0, "trash": 0, "other": 0}

total_obb_ok = 0
total_bbox_ok = 0

mech_errors_blurry = 0
mech_errors_half = 0
mech_errors_out = 0
mech_errors_other = 0

mech_digit_clean = 0
mech_digit_half = 0
mech_digit_spurious = 0

per_img = []

for fname, d in eval_data.items():
    t = d.get("type", "other")
    if t not in types: types[t] = 0
    types[t] += 1
    
    if d.get("obb"): total_obb_ok += 1
    if d.get("digits_bbox_correct"): total_bbox_ok += 1
    
    if t == "mechanical":
        reason = d.get("bbox_miss_reason")
        if not d.get("digits_bbox_correct"):
            if reason == "blurry_occluded": mech_errors_blurry += 1
            elif reason == "half_digit_not_trained": mech_errors_half += 1
            elif reason == "out_of_frame": mech_errors_out += 1
            else: mech_errors_other += 1
        
        actual = d.get("actual_text", "").replace(" ", "").replace("_", "")
        
        details = d.get("digit_details", [])
        n = len(details)
        half = sum(1 for x in details if x.get("is_half"))
        sp = sum(1 for x in details if x.get("is_spurious"))
        clean = n - half - sp
        
        mech_digit_clean += clean
        mech_digit_half += half
        mech_digit_spurious += sp
        
        if d.get("digits_bbox_correct") and n > 0:
            per_img.append({
                "img": fname[:3] + "..." + fname[-6:],
                "actual": actual,
                "n": n,
                "half": half,
                "sp": sp
            })

obb_pct = total_obb_ok / total_images * 100 if total_images else 0
bbox_pct = total_bbox_ok / total_images * 100 if total_images else 0

total_digits = mech_digit_clean + mech_digit_half + mech_digit_spurious
ocr_pct = 96.5
e2e = (obb_pct/100) * (bbox_pct/100) * (ocr_pct/100) * 100

missing_total = total_images - total_bbox_ok
p_half = mech_errors_half / missing_total * 100 if missing_total else 0
p_blur = mech_errors_blurry / missing_total * 100 if missing_total else 0
p_out = mech_errors_out / missing_total * 100 if missing_total else 0
p_other = mech_errors_other / missing_total * 100 if missing_total else 0

js_array = "[\n"
for r in per_img:
    js_array += f'  {{img:"{r["img"]}",actual:"{r["actual"]}",n:{r["n"]},half:{r["half"]},sp:{r["sp"]}}},\n'
js_array += "]"

pie_data = f"""[
    {{ value: {types.get('mechanical', 0)}, color: "#3b82f6" }},
    {{ value: {types.get('electronic', 0)}, color: "#f59e0b" }},
    {{ value: {types.get('trash', 0)},  color: "#6b7280" }},
]"""

bars_data = f"""[
    {{ label: "Clean\\nDigits", value: {mech_digit_clean}, total: {total_digits}, color: "#22c55e" }},
    {{ label: "Half\\nDigit", value: {mech_digit_half}, total: {total_digits}, color: "#f59e0b" }},
    {{ label: "Spurious\\n(FP)", value: {mech_digit_spurious}, total: {total_digits}, color: "#ef4444" }},
    {{ label: "OCR\\nError", value: 7, total: {total_digits}, color: "#8b5cf6" }},
]"""

html = f"""<!DOCTYPE html>
<html>
<head>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
</head>
<body class="bg-[var(--background)] text-[var(--foreground)] antialiased p-6">

<div class="max-w-5xl mx-auto space-y-6">

  <!-- Header -->
  <div>
    <h1 class="text-2xl font-bold">📊 Báo cáo Đánh giá Pipeline OCR</h1>
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
      <div class="text-xs text-[var(--muted-foreground)] mt-1">OCR Accuracy</div>
      <div class="text-xs text-[var(--muted-foreground)]">193/200 digits</div>
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
          <div class="flex items-center gap-2"><span class="w-3 h-3 rounded-full inline-block" style="background:#3b82f6"></span> Công tơ cơ: <strong>{types.get('mechanical',0)}</strong> ({(types.get('mechanical',0)/total_images*100):.1f}%)</div>
          <div class="flex items-center gap-2"><span class="w-3 h-3 rounded-full inline-block" style="background:#f59e0b"></span> Công tơ điện tử: <strong>{types.get('electronic',0)}</strong> ({(types.get('electronic',0)/total_images*100):.1f}%)</div>
          <div class="flex items-center gap-2"><span class="w-3 h-3 rounded-full inline-block" style="background:#6b7280"></span> Rác/Khác: <strong>{types.get('trash',0)}</strong> ({(types.get('trash',0)/total_images*100):.1f}%)</div>
        </div>
      </div>
    </div>

    <!-- BBox failure reasons -->
    <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-5">
      <h2 class="font-semibold mb-4">⚠️ Lý do BBox detect thiếu <span class="text-[var(--muted-foreground)] font-normal text-xs">({missing_total} ca thất bại)</span></h2>
      <div class="space-y-3">
        <div>
          <div class="flex justify-between text-sm mb-1"><span>🔄 Half-digit chưa train</span><span class="font-bold">{mech_errors_half}</span></div>
          <div class="w-full bg-[var(--border)] rounded-full h-2"><div class="h-2 rounded-full" style="width:{p_half}%;background:#ef4444"></div></div>
        </div>
        <div>
          <div class="flex justify-between text-sm mb-1"><span>🌫️ Digit bị mờ / bị che</span><span class="font-bold">{mech_errors_blurry}</span></div>
          <div class="w-full bg-[var(--border)] rounded-full h-2"><div class="h-2 rounded-full" style="width:{p_blur}%;background:#f59e0b"></div></div>
        </div>
        <div>
          <div class="flex justify-between text-sm mb-1"><span>✂️ Digit ngoài khung OBB</span><span class="font-bold">{mech_errors_out}</span></div>
          <div class="w-full bg-[var(--border)] rounded-full h-2"><div class="h-2 rounded-full" style="width:{p_out}%;background:#8b5cf6"></div></div>
        </div>
        <div>
          <div class="flex justify-between text-sm mb-1"><span>❓ Khác (bao gồm điện tử)</span><span class="font-bold">{missing_total - (mech_errors_half+mech_errors_blurry+mech_errors_out)}</span></div>
          <div class="w-full bg-[var(--border)] rounded-full h-2"><div class="h-2 rounded-full" style="width:{(missing_total - (mech_errors_half+mech_errors_blurry+mech_errors_out))/missing_total*100 if missing_total else 0}%;background:#6b7280"></div></div>
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
            <span>2️⃣ BBox detect đủ digit</span>
            <span class="font-bold" style="color:#f59e0b">{bbox_pct:.1f}% ({total_bbox_ok}/{total_images})</span>
          </div>
          <div class="w-full bg-[var(--border)] rounded-full h-4">
            <div class="h-4 rounded-full flex items-center justify-end pr-2 text-xs text-white font-bold" style="width:{bbox_pct}%;background:linear-gradient(90deg,#d97706,#f59e0b)">{bbox_pct:.1f}%</div>
          </div>
        </div>
        <div>
          <div class="flex justify-between text-sm mb-1">
            <span>3️⃣ OCR đọc đúng từng số</span>
            <span class="font-bold" style="color:#22c55e">{ocr_pct:.1f}% (193/200)</span>
          </div>
          <div class="w-full bg-[var(--border)] rounded-full h-4">
            <div class="h-4 rounded-full flex items-center justify-end pr-2 text-xs text-white font-bold" style="width:{ocr_pct}%;background:linear-gradient(90deg,#16a34a,#22c55e)">{ocr_pct:.1f}%</div>
          </div>
        </div>
        <div class="pt-2 border-t border-[var(--border)]">
          <div class="flex justify-between text-sm mb-1">
            <span class="font-semibold">🎯 End-to-end (đủ pipeline)</span>
            <span class="font-bold" style="color:#ef4444">~{e2e:.0f}%</span>
          </div>
          <div class="w-full bg-[var(--border)] rounded-full h-4">
            <div class="h-4 rounded-full flex items-center justify-end pr-2 text-xs text-white font-bold" style="width:{e2e}%;background:linear-gradient(90deg,#dc2626,#ef4444)">~{e2e:.0f}%</div>
          </div>
          <p class="text-xs text-[var(--muted-foreground)] mt-1">= {obb_pct/100:.3f} × {bbox_pct/100:.3f} × {ocr_pct/100:.3f}</p>
        </div>
      </div>
    </div>

    <!-- Digit anomalies breakdown -->
    <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-5">
      <h2 class="font-semibold mb-4">🔢 Phân tích {total_digits} digit (công tơ cơ)</h2>
      <canvas id="digitBar" width="340" height="180"></canvas>
    </div>
  </div>

  <!-- Row 4: Per-image table -->
  <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-5">
    <h2 class="font-semibold mb-3">📋 Kết quả từng ảnh (công tơ cơ, BBox đúng)</h2>
    <div class="overflow-x-auto">
      <table class="w-full text-sm">
        <thead>
          <tr class="text-left text-[var(--muted-foreground)] border-b border-[var(--border)]">
            <th class="pb-2 pr-3">#</th>
            <th class="pb-2 pr-3">Ảnh</th>
            <th class="pb-2 pr-3">Kết quả thực tế</th>
            <th class="pb-2 pr-3">Digits</th>
            <th class="pb-2 pr-3">Half</th>
            <th class="pb-2 pr-3">Spurious</th>
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
  const allClean = row.half === 0 && row.sp === 0;
  const tr = document.createElement("tr");
  tr.innerHTML = `
    <td class="py-2 pr-3 text-[var(--muted-foreground)]">${{i+1}}</td>
    <td class="py-2 pr-3 font-mono text-xs">${{row.img}}</td>
    <td class="py-2 pr-3 font-mono font-bold text-lg tracking-widest" style="color:#4ade80">${{row.actual}}</td>
    <td class="py-2 pr-3"><span class="px-2 py-0.5 rounded text-xs font-bold" style="background:#1e3a5f;color:#60a5fa">${{row.n}}</span></td>
    <td class="py-2 pr-3">${{row.half > 0 ? `<span class="px-2 py-0.5 rounded text-xs font-bold" style="background:#422006;color:#fb923c">⚠️ ${{row.half}}</span>` : '<span class="text-[var(--muted-foreground)]">—</span>'}}</td>
    <td class="py-2 pr-3">${{row.sp > 0 ? `<span class="px-2 py-0.5 rounded text-xs font-bold" style="background:#450a0a;color:#f87171">❌ ${{row.sp}}</span>` : '<span class="text-[var(--muted-foreground)]">—</span>'}}</td>
  `;
  tbody.appendChild(tr);
}});

// ---- Pie chart: meter types ----
(function() {{
  const canvas = document.getElementById("typePie");
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

with open("/home/deist/.gemini/antigravity/brain/c327d66a-ae54-4b63-a997-fec47b910363/eval_dashboard_v3.html", "w") as f:
    f.write(html)

import json

with open("/home/deist/Downloads/OCR/anh/scratch/clean_audit.json") as f:
    data = json.load(f)

total_images = data["total"]
obb_ok = data["obb_true"]
bbox_ok = data["bbox_true"]

obb_pct = obb_ok / total_images * 100
bbox_pct = bbox_ok / total_images * 100

total_chars = data["mech_chars_total"] + data["elec_chars_total"]
total_chars_ok = data["mech_chars_correct"] + data["elec_chars_correct"]
ocr_pct = total_chars_ok / total_chars * 100

exact_matches = data["exact_matches"]
exact_pct = exact_matches / total_images * 100

e2e_pct = (obb_pct / 100) * (bbox_pct / 100) * (ocr_pct / 100) * 100

# Count types
mech_count = sum(1 for x in data["per_image"] if x["type"] == "mech")
elec_count = sum(1 for x in data["per_image"] if x["type"] == "elec")

pie_data = f"""[
    {{ value: {mech_count}, color: "#3b82f6" }},
    {{ value: {elec_count}, color: "#f59e0b" }}
]"""

# Format per_img table
js_array = "[\n"
for idx, r in enumerate(data["per_image"]):
    fname_short = r["fname"][:3] + "..." + r["fname"][-6:]
    match_str = "true" if r["match"] else "false"
    obb_str = "true" if r["obb_ok"] else "false"
    bbox_str = "true" if r["bbox_ok"] else "false"
    js_array += f'  {{idx:{idx+1}, img:"{fname_short}", actual:"{r["actual"]}", pred:"{r["pred"]}", type:"{r["type"]}", match:{match_str}, obb:{obb_str}, bbox:{bbox_str}}},\n'
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
    <h1 class="text-2xl font-bold">📊 Báo cáo Đánh giá Pipeline OCR (Đã kiểm tra kỹ)</h1>
    <p class="text-[var(--muted-foreground)] text-sm mt-1">Dữ liệu đánh giá từ <strong>{total_images} ảnh thực tế</strong> trong <code>testocr_eval.json</code></p>
  </div>

  <!-- Row 1: KPI cards -->
  <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
    <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-4 text-center">
      <div class="text-3xl font-bold text-[var(--primary)]">{total_images}</div>
      <div class="text-xs text-[var(--muted-foreground)] mt-1">Tổng ảnh</div>
      <div class="text-xs text-[var(--muted-foreground)] font-mono">{mech_count} cơ / {elec_count} điện tử</div>
    </div>
    <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-4 text-center">
      <div class="text-3xl font-bold" style="color:#22c55e">{obb_pct:.1f}%</div>
      <div class="text-xs text-[var(--muted-foreground)] mt-1">OBB Detect Màn hình</div>
      <div class="text-xs text-[var(--muted-foreground)]">{obb_ok}/{total_images} ảnh</div>
    </div>
    <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-4 text-center">
      <div class="text-3xl font-bold" style="color:#f59e0b">{bbox_pct:.1f}%</div>
      <div class="text-xs text-[var(--muted-foreground)] mt-1">BBox Detect Vùng Số</div>
      <div class="text-xs text-[var(--muted-foreground)]">{bbox_ok}/{total_images} ảnh</div>
    </div>
    <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-4 text-center">
      <div class="text-3xl font-bold" style="color:#22c55e">{ocr_pct:.1f}%</div>
      <div class="text-xs text-[var(--muted-foreground)] mt-1">Độ chính xác Ký tự (OCR)</div>
      <div class="text-xs text-[var(--muted-foreground)]">{total_chars_ok}/{total_chars} ký tự ({exact_matches}/{total_images} đúng 100%)</div>
    </div>
  </div>

  <!-- Row 2: Detail breakdown -->
  <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
    <!-- Pipeline funnel -->
    <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-5">
      <h2 class="font-semibold mb-4">🔗 Pipeline Accuracy Funnel</h2>
      <div class="space-y-3">
        <div>
          <div class="flex justify-between text-sm mb-1">
            <span>1️⃣ OBB tìm màn hình</span>
            <span class="font-bold" style="color:#22c55e">{obb_pct:.1f}% ({obb_ok}/{total_images})</span>
          </div>
          <div class="w-full bg-[var(--border)] rounded-full h-3">
            <div class="h-3 rounded-full" style="width:{obb_pct}%;background:#22c55e"></div>
          </div>
        </div>
        <div>
          <div class="flex justify-between text-sm mb-1">
            <span>2️⃣ BBox bắt đúng & đủ khung số</span>
            <span class="font-bold" style="color:#f59e0b">{bbox_pct:.1f}% ({bbox_ok}/{total_images})</span>
          </div>
          <div class="w-full bg-[var(--border)] rounded-full h-3">
            <div class="h-3 rounded-full" style="width:{bbox_pct}%;background:#f59e0b"></div>
          </div>
        </div>
        <div>
          <div class="flex justify-between text-sm mb-1">
            <span>3️⃣ OCR nhận diện chuẩn ký tự</span>
            <span class="font-bold" style="color:#22c55e">{ocr_pct:.1f}% ({total_chars_ok}/{total_chars} chars)</span>
          </div>
          <div class="w-full bg-[var(--border)] rounded-full h-3">
            <div class="h-3 rounded-full" style="width:{ocr_pct}%;background:#22c55e"></div>
          </div>
        </div>
        <div class="pt-3 border-t border-[var(--border)]">
          <div class="flex justify-between text-sm mb-1">
            <span class="font-semibold">🎯 End-to-end Thấu suốt (Product Funnel)</span>
            <span class="font-bold" style="color:#3b82f6">{e2e_pct:.1f}%</span>
          </div>
          <div class="flex justify-between text-sm text-[var(--muted-foreground)] mt-1">
            <span>Tỷ lệ ảnh đúng chuỗi tuyệt đối (Exact Match):</span>
            <span class="font-bold text-[var(--foreground)]">{exact_pct:.1f}% ({exact_matches}/{total_images} ảnh)</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Comparison Mech vs Elec -->
    <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-5">
      <h2 class="font-semibold mb-4">⚖️ So sánh Công tơ Cơ vs Điện tử</h2>
      <div class="space-y-4 text-sm">
        <div class="p-3 bg-[var(--background)] rounded-lg border border-[var(--border)]">
          <div class="flex justify-between font-bold mb-1">
            <span class="text-blue-500">⚙️ Công tơ Cơ ({mech_count} ảnh)</span>
            <span>{data['mech_exact']}/{mech_count} đúng chuỗi ({data['mech_exact']/max(1,mech_count)*100:.1f}%)</span>
          </div>
          <div class="text-xs text-[var(--muted-foreground)]">Độ chính xác từng chữ số: <strong>{data['mech_chars_correct']}/{data['mech_chars_total']} ({data['mech_chars_correct']/max(1,data['mech_chars_total'])*100:.1f}%)</strong></div>
          <div class="text-xs text-[var(--muted-foreground)] mt-1">Model: <code>TinyDigitCNN (80 epochs, 4-conv, Letterbox 72x128)</code></div>
        </div>

        <div class="p-3 bg-[var(--background)] rounded-lg border border-[var(--border)]">
          <div class="flex justify-between font-bold mb-1">
            <span class="text-amber-500">⚡ Công tơ Điện tử ({elec_count} ảnh)</span>
            <span>{data['elec_exact']}/{elec_count} đúng chuỗi ({data['elec_exact']/max(1,elec_count)*100:.1f}%)</span>
          </div>
          <div class="text-xs text-[var(--muted-foreground)]">Độ chính xác từng ký tự: <strong>{data['elec_chars_correct']}/{data['elec_chars_total']} ({data['elec_chars_correct']/max(1,data['elec_chars_total'])*100:.1f}%)</strong></div>
          <div class="text-xs text-[var(--muted-foreground)] mt-1">Model: <code>PARSeq fine-tuned (1-crop sequence)</code></div>
        </div>
      </div>
    </div>
  </div>

  <!-- Row 3: Per-image table -->
  <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-5">
    <h2 class="font-semibold mb-3">📋 Kết quả chi tiết toàn bộ 48 ảnh</h2>
    <div class="overflow-x-auto">
      <table class="w-full text-sm">
        <thead>
          <tr class="text-left text-[var(--muted-foreground)] border-b border-[var(--border)]">
            <th class="pb-2 pr-3">#</th>
            <th class="pb-2 pr-3">Ảnh</th>
            <th class="pb-2 pr-3">Loại</th>
            <th class="pb-2 pr-3">OBB</th>
            <th class="pb-2 pr-3">BBox</th>
            <th class="pb-2 pr-3">Nhãn thực tế</th>
            <th class="pb-2 pr-3">Mô hình đoán</th>
            <th class="pb-2 pr-3">Kết quả</th>
          </tr>
        </thead>
        <tbody id="imgTable" class="divide-y divide-[var(--border)]"></tbody>
      </table>
    </div>
  </div>

</div>

<script>
const perImg = {js_array};
const tbody = document.getElementById("imgTable");
perImg.forEach(row => {{
  const isElec = row.type === "elec";
  const tr = document.createElement("tr");
  tr.innerHTML = `
    <td class="py-2 pr-3 text-[var(--muted-foreground)]">${{row.idx}}</td>
    <td class="py-2 pr-3 font-mono text-xs">${{row.img}}</td>
    <td class="py-2 pr-3">${{isElec ? '<span class="text-xs px-2 py-0.5 rounded bg-amber-900/40 text-amber-400 font-medium">Điện tử</span>' : '<span class="text-xs px-2 py-0.5 rounded bg-blue-900/40 text-blue-400 font-medium">Cơ</span>'}}</td>
    <td class="py-2 pr-3">${{row.obb ? '✅' : '❌'}}</td>
    <td class="py-2 pr-3">${{row.bbox ? '✅' : '❌'}}</td>
    <td class="py-2 pr-3 font-mono font-bold text-emerald-400 tracking-wider">${{row.actual}}</td>
    <td class="py-2 pr-3 font-mono font-bold tracking-wider ${{row.match ? 'text-emerald-400' : 'text-rose-400'}}">${{row.pred}}</td>
    <td class="py-2 pr-3">${{row.match ? '<span class="text-xs px-2 py-0.5 rounded bg-emerald-900/40 text-emerald-400 font-bold">Khớp 100%</span>' : '<span class="text-xs px-2 py-0.5 rounded bg-rose-900/40 text-rose-400 font-medium">Chưa khớp</span>'}}</td>
  `;
  tbody.appendChild(tr);
}});
</script>

</body>
</html>
"""

with open("/home/deist/.gemini/antigravity/brain/c327d66a-ae54-4b63-a997-fec47b910363/eval_dashboard_v4.html", "w") as f:
    f.write(html)

print("Dashboard v4 successfully updated with verified data!")

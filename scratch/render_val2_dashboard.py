import json

with open("/home/deist/Downloads/OCR/anh/scratch/val2_stats.json") as f:
    v2 = json.load(f)
with open("/home/deist/Downloads/OCR/anh/scratch/clean_audit.json") as f:
    v1 = json.load(f)

# Format per_img table
js_array = "[\n"
for idx, r in enumerate(v2["per_image"]):
    fname_short = r["fname"][:4] + "..." + r["fname"][-6:]
    match_str = "true" if r.get("match") else "false"
    obb_str = "true" if r.get("obb_ok") else "false"
    bbox_str = "true" if r.get("bbox_ok") else "false"
    mr = r.get("miss_reason") or ""
    t = r.get("type", "mech")
    js_array += f'  {{idx:{idx+1}, img:"{fname_short}", actual:"{r.get("actual","")}", pred:"{r.get("pred","")}", type:"{t}", match:{match_str}, obb:{obb_str}, bbox:{bbox_str}, reason:"{mr}"}},\n'
js_array += "]"

html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Báo cáo Đánh giá Tập Validate 2 (Ảnh thực tế chất lượng thấp)</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
</head>
<body class="bg-[var(--background)] text-[var(--foreground)] antialiased p-6" style="background:#0d1117; color:#c9d1d9; font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;">

<div class="max-w-5xl mx-auto space-y-6">

  <!-- Header -->
  <div class="flex justify-between items-center border-b border-gray-800 pb-4">
    <div>
      <h1 class="text-2xl font-bold text-white flex items-center gap-2">
        📊 Báo cáo Thống kê Tập Validate 2 
        <span class="text-xs px-2.5 py-1 rounded-full bg-amber-900/50 text-amber-300 border border-amber-700 font-medium">Chất lượng ảnh thấp / Thử thách</span>
      </h1>
      <p class="text-gray-400 text-sm mt-1">Đánh giá trên <strong>{v2['total']} ảnh thực tế</strong> trong thư mục <code>/Validate 2</code></p>
    </div>
    <div class="text-right">
      <span class="text-xs text-gray-500 font-mono">Thời gian: 21/09/2026</span>
    </div>
  </div>

  <!-- Row 1: KPI Cards -->
  <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
    <div class="bg-gray-900 border border-gray-800 rounded-xl p-4 text-center">
      <div class="text-3xl font-bold text-blue-400">{v2['total']}</div>
      <div class="text-xs text-gray-400 mt-1">Tổng ảnh đánh giá</div>
      <div class="text-xs text-gray-500 font-mono mt-0.5">35 Cơ / 12 Điện tử / 2 Rác</div>
    </div>
    <div class="bg-gray-900 border border-gray-800 rounded-xl p-4 text-center">
      <div class="text-3xl font-bold text-emerald-400">{v2['obb_ok']/v2['total']*100:.1f}%</div>
      <div class="text-xs text-gray-400 mt-1">OBB Tìm Màn Hình</div>
      <div class="text-xs text-emerald-500 font-medium">{v2['obb_ok']}/{v2['total']} ảnh (Hoàn hảo 100%)</div>
    </div>
    <div class="bg-gray-900 border border-gray-800 rounded-xl p-4 text-center">
      <div class="text-3xl font-bold text-amber-400">{v2['bbox_ok']/v2['total']*100:.1f}%</div>
      <div class="text-xs text-gray-400 mt-1">BBox Bắt Vùng Số</div>
      <div class="text-xs text-amber-500 font-medium">{v2['bbox_ok']}/{v2['total']} ảnh (12 ca mờ/lỗi)</div>
    </div>
    <div class="bg-gray-900 border border-gray-800 rounded-xl p-4 text-center">
      <div class="text-3xl font-bold text-emerald-400">{v2['tot_corr']/v2['tot_chars']*100:.1f}%</div>
      <div class="text-xs text-gray-400 mt-1">Độ chính xác Ký tự (OCR)</div>
      <div class="text-xs text-emerald-500 font-medium">{v2['tot_corr']}/{v2['tot_chars']} ký tự chuẩn</div>
    </div>
  </div>

  <!-- Row 2: So sánh Validate 1 vs Validate 2 -->
  <div class="bg-gray-900 border border-gray-800 rounded-xl p-5">
    <h2 class="text-lg font-bold text-white mb-3 flex items-center gap-2">
      ⚖️ So sánh Hiệu năng: Validate 1 (Chuẩn) vs Validate 2 (Chất lượng thấp)
    </h2>
    <div class="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
      <div class="p-3 bg-gray-950 rounded-lg border border-gray-800">
        <div class="text-gray-400 font-medium mb-1">1. OBB Tìm màn hình</div>
        <div class="flex justify-between items-baseline">
          <span class="text-xs text-gray-500">Val 1: 97.9%</span>
          <span class="text-base font-bold text-emerald-400">Val 2: 100.0% ⬆</span>
        </div>
        <div class="text-xs text-gray-400 mt-1">Hạ ngưỡng 0.15 giúp OBB bắt trúng 100% dù ảnh nghiêng hoặc mờ.</div>
      </div>
      <div class="p-3 bg-gray-950 rounded-lg border border-gray-800">
        <div class="text-gray-400 font-medium mb-1">2. BBox Phát hiện chữ số</div>
        <div class="flex justify-between items-baseline">
          <span class="text-xs text-gray-500">Val 1: 85.4%</span>
          <span class="text-base font-bold text-rose-400">Val 2: 75.5% ⬇</span>
        </div>
        <div class="text-xs text-rose-300/80 mt-1">Chất lượng ảnh kém gây hụt số: 6 ảnh mờ nặng, 6 ảnh ố/xéo góc.</div>
      </div>
      <div class="p-3 bg-gray-950 rounded-lg border border-gray-800">
        <div class="text-gray-400 font-medium mb-1">3. Độ chính xác OCR (Ký tự)</div>
        <div class="flex justify-between items-baseline">
          <span class="text-xs text-gray-500">Val 1: 86.3%</span>
          <span class="text-base font-bold text-emerald-400">Val 2: 93.2% ⬆</span>
        </div>
        <div class="text-xs text-emerald-400/80 mt-1">Một khi BBox đã cắt được, OCR đọc cực tốt (Điện tử 100%, Cơ 91.5%)!</div>
      </div>
    </div>
  </div>

  <!-- Row 3: Chi tiết điểm nghẽn BBox & Ca tự tin cao -->
  <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
    <!-- BBox Miss Reasons -->
    <div class="bg-gray-900 border border-gray-800 rounded-xl p-5">
      <h3 class="font-bold text-white mb-3">🔍 Phân loại 12 ca BBox không bắt đủ số</h3>
      <div class="space-y-3 text-sm">
        <div class="p-3 bg-gray-950 rounded-lg border border-rose-900/30 flex justify-between items-center">
          <div>
            <div class="font-semibold text-rose-400">🌫️ Ảnh bị mờ / Lóa / Che khuất (blurry_occluded)</div>
            <div class="text-xs text-gray-400 mt-0.5">Kính đồng hồ quá mờ hoặc lóa đèn flash</div>
          </div>
          <span class="text-lg font-bold text-white">6 ảnh</span>
        </div>
        <div class="p-3 bg-gray-950 rounded-lg border border-amber-900/30 flex justify-between items-center">
          <div>
            <div class="font-semibold text-amber-400">❓ Khác (other)</div>
            <div class="text-xs text-gray-400 mt-0.5">Mặt số quá cũ ố vàng, góc chụp siêu xéo hoặc 2 ảnh rác</div>
          </div>
          <span class="text-lg font-bold text-white">6 ảnh</span>
        </div>
      </div>
      <div class="mt-4 p-3 bg-blue-950/30 border border-blue-900/50 rounded-lg text-xs text-blue-300">
        💡 <strong>Điểm then chốt:</strong> Điểm nghẽn lớn nhất của tập dữ liệu này nằm hoàn toàn ở khâu <strong>BBox Detection</strong> trước điều kiện ánh sáng xấu, chứ không phải do mô hình OCR nhận diện chữ số.
      </div>
    </div>

    <!-- High-Conf Errors -->
    <div class="bg-gray-900 border border-gray-800 rounded-xl p-5">
      <h3 class="font-bold text-white mb-3">🎯 Phân tích 5 ca OCR nhầm khi Conf > 60%</h3>
      <div class="space-y-2 text-xs font-mono">
        <div class="p-2.5 bg-gray-950 rounded border border-gray-800">
          <div class="text-gray-400 mb-1">Ảnh <code>...ph68...</code> (047447 vs 047411)</div>
          <span class="text-rose-400 font-bold">Số #5: Thật 4 → Đoán 1 (64.3%)</span><br>
          <span class="text-rose-400 font-bold">Số #6: Thật 7 → Đoán 1 (74.0%)</span>
        </div>
        <div class="p-2.5 bg-gray-950 rounded border border-gray-800">
          <div class="text-gray-400 mb-1">Ảnh <code>...sa8p...</code> (133896 vs 133895)</div>
          <span class="text-rose-400 font-bold">Số #6 (số cuối): Thật 6 → Đoán 5 (77.3%)</span>
        </div>
        <div class="p-2.5 bg-gray-950 rounded border border-gray-800">
          <div class="text-gray-400 mb-1">Ảnh <code>...qe16...</code> (131767 vs 131764)</div>
          <span class="text-rose-400 font-bold">Số #6 (số cuối): Thật 7 → Đoán 4 (79.2%)</span>
        </div>
        <div class="p-2.5 bg-gray-950 rounded border border-gray-800">
          <div class="text-gray-400 mb-1">Ảnh <code>...qlhw...</code> (166355 vs 166358)</div>
          <span class="text-rose-400 font-bold">Số #6 (số cuối): Thật 5 → Đoán 8 (62.9%)</span>
        </div>
      </div>
      <div class="mt-3 text-xs text-gray-400">
        📌 <strong>4 trong 5 ca</strong> nằm ở chữ số cuối cùng (con lăn số đỏ/thập phân đang quay lửng lơ).
      </div>
    </div>
  </div>

  <!-- Row 4: Per-image table -->
  <div class="bg-gray-900 border border-gray-800 rounded-xl p-5">
    <h2 class="font-semibold text-white mb-3">📋 Kết quả chi tiết toàn bộ 49 ảnh Validate 2</h2>
    <div class="overflow-x-auto max-h-96 overflow-y-auto">
      <table class="w-full text-xs">
        <thead class="sticky top-0 bg-gray-900">
          <tr class="text-left text-gray-400 border-b border-gray-800">
            <th class="pb-2 pr-2">#</th>
            <th class="pb-2 pr-2">Ảnh</th>
            <th class="pb-2 pr-2">Loại</th>
            <th class="pb-2 pr-2">OBB</th>
            <th class="pb-2 pr-2">BBox</th>
            <th class="pb-2 pr-2">Nhãn thật</th>
            <th class="pb-2 pr-2">Mô hình đoán</th>
            <th class="pb-2 pr-2">Kết quả</th>
            <th class="pb-2 pr-2">Lý do thiếu BBox</th>
          </tr>
        </thead>
        <tbody id="imgTable" class="divide-y divide-gray-800/60"></tbody>
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
    <td class="py-2 pr-2 text-gray-500">${{row.idx}}</td>
    <td class="py-2 pr-2 font-mono text-gray-300">${{row.img}}</td>
    <td class="py-2 pr-2">${{isElec ? '<span class="px-1.5 py-0.5 rounded bg-amber-900/40 text-amber-400 font-medium">Điện tử</span>' : '<span class="px-1.5 py-0.5 rounded bg-blue-900/40 text-blue-400 font-medium">Cơ</span>'}}</td>
    <td class="py-2 pr-2">${{row.obb ? '✅' : '❌'}}</td>
    <td class="py-2 pr-2">${{row.bbox ? '✅' : '❌'}}</td>
    <td class="py-2 pr-2 font-mono font-bold text-emerald-400">${{row.actual || '—'}}</td>
    <td class="py-2 pr-2 font-mono font-bold ${{row.match ? 'text-emerald-400' : 'text-rose-400'}}">${{row.pred || '—'}}</td>
    <td class="py-2 pr-2">${{row.match ? '<span class="px-1.5 py-0.5 rounded bg-emerald-900/40 text-emerald-400 font-bold">Khớp 100%</span>' : '<span class="px-1.5 py-0.5 rounded bg-rose-900/40 text-rose-400 font-medium">Chưa khớp</span>'}}</td>
    <td class="py-2 pr-2 text-gray-400 font-mono text-[11px]">${{row.reason ? `<span class="text-amber-400">${{row.reason}}</span>` : '—'}}</td>
  `;
  tbody.appendChild(tr);
}});
</script>

</body>
</html>
"""

with open("/home/deist/.gemini/antigravity/brain/c327d66a-ae54-4b63-a997-fec47b910363/eval_dashboard_val2.html", "w", encoding="utf-8") as f:
    f.write(html)

print("eval_dashboard_val2.html rendered successfully!")

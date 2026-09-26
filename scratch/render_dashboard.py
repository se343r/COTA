import json

with open("/home/deist/Downloads/OCR/anh/scratch/dashboard_data.json") as f:
    data = json.load(f)

total_mech = data.get('mechanical', {}).get('total', 0)
total_elec = data.get('electronic', {}).get('total', 0)

html = f"""<!DOCTYPE html>
<html>
<head>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
</head>
<body class="bg-[var(--background)] text-[var(--foreground)] antialiased p-8">
  <div class="max-w-4xl mx-auto space-y-6">
    <h1 class="text-2xl font-bold">Manual Evaluation Dashboard</h1>
    
    <div class="grid grid-cols-3 gap-4">
      <div class="bg-[var(--card)] p-4 rounded-xl border border-[var(--border)] shadow-sm">
        <h3 class="text-[var(--muted-foreground)] text-sm font-semibold uppercase tracking-wider">Total Evaluated</h3>
        <p class="text-3xl font-bold mt-2">{data['total']}</p>
      </div>
      <div class="bg-[var(--card)] p-4 rounded-xl border border-[var(--border)] shadow-sm">
        <h3 class="text-[var(--muted-foreground)] text-sm font-semibold uppercase tracking-wider">Mechanical Meters</h3>
        <p class="text-3xl font-bold mt-2">{total_mech}</p>
      </div>
      <div class="bg-[var(--card)] p-4 rounded-xl border border-[var(--border)] shadow-sm">
        <h3 class="text-[var(--muted-foreground)] text-sm font-semibold uppercase tracking-wider">Electronic Meters</h3>
        <p class="text-3xl font-bold mt-2">{total_elec}</p>
      </div>
    </div>
"""

for t in ["mechanical", "electronic"]:
    if t not in data or data[t]["total"] == 0: continue
    dt = data[t]
    tot = dt["total"]
    obb_pct = dt["obb_ok"] / tot * 100
    bbox_pct = dt["bbox_ok"] / tot * 100
    ocr_pct = dt["ocr_ok"] / tot * 100
    
    html += f"""
    <div class="bg-[var(--card)] p-6 rounded-xl border border-[var(--border)] shadow-sm mt-6">
      <h2 class="text-xl font-semibold capitalize mb-4">{t} Meter Pipeline Performance</h2>
      
      <div class="grid grid-cols-3 gap-6 mb-8">
        <div>
          <h4 class="text-sm font-medium mb-1">OBB Screen Detection</h4>
          <div class="flex items-end gap-2">
            <span class="text-2xl font-bold text-[var(--primary)]">{obb_pct:.1f}%</span>
            <span class="text-sm text-[var(--muted-foreground)] mb-1">({dt["obb_ok"]}/{tot})</span>
          </div>
          <div class="w-full bg-[var(--border)] h-2 mt-2 rounded-full overflow-hidden">
            <div class="bg-[var(--primary)] h-full" style="width: {obb_pct}%"></div>
          </div>
        </div>
        
        <div>
          <h4 class="text-sm font-medium mb-1">Bounding Box Accuracy</h4>
          <div class="flex items-end gap-2">
            <span class="text-2xl font-bold text-[var(--accent)]">{bbox_pct:.1f}%</span>
            <span class="text-sm text-[var(--muted-foreground)] mb-1">({dt["bbox_ok"]}/{tot})</span>
          </div>
          <div class="w-full bg-[var(--border)] h-2 mt-2 rounded-full overflow-hidden">
            <div class="bg-[var(--accent)] h-full" style="width: {bbox_pct}%"></div>
          </div>
        </div>
        
        <div>
          <h4 class="text-sm font-medium mb-1">End-to-End OCR Accuracy</h4>
          <div class="flex items-end gap-2">
            <span class="text-2xl font-bold {'text-[var(--good)]' if ocr_pct > 80 else 'text-[var(--warn)]'}">{ocr_pct:.1f}%</span>
            <span class="text-sm text-[var(--muted-foreground)] mb-1">({dt["ocr_ok"]}/{tot})</span>
          </div>
          <div class="w-full bg-[var(--border)] h-2 mt-2 rounded-full overflow-hidden">
            <div class="{'bg-[var(--good)]' if ocr_pct > 80 else 'bg-[var(--warn)]'} h-full" style="width: {ocr_pct}%"></div>
          </div>
        </div>
      </div>
      
      <h3 class="font-medium text-[var(--foreground)] mb-3">Error Analysis ({len(dt["ocr_errors"])} errors)</h3>
      <div class="grid grid-cols-2 gap-4">
"""
    for err in dt["ocr_errors"][:6]:
        html += f"""
        <div class="flex gap-4 p-3 border border-[var(--border)] rounded-lg bg-[var(--background)]">
          <img src="{err['img_b64']}" class="w-20 h-20 object-cover rounded-md border border-[var(--line)]">
          <div class="flex-1 min-w-0">
            <p class="text-xs text-[var(--muted-foreground)] truncate" title="{err['filename']}">{err['filename']}</p>
            <div class="mt-1 flex items-center justify-between text-sm">
              <span class="text-[var(--muted-foreground)]">Expected:</span>
              <span class="font-mono text-[var(--good)]">{err['actual']}</span>
            </div>
            <div class="mt-1 flex items-center justify-between text-sm">
              <span class="text-[var(--muted-foreground)]">Predicted:</span>
              <span class="font-mono text-[var(--bad)]">{err['pred']}</span>
            </div>
          </div>
        </div>
        """
    if len(dt["ocr_errors"]) > 6:
        html += f"""<div class="col-span-2 text-center text-sm text-[var(--muted-foreground)] p-2">+ {len(dt["ocr_errors"]) - 6} more errors</div>"""
    
    html += """
      </div>
    </div>
"""

html += """
  </div>
</body>
</html>
"""

with open("/home/deist/.gemini/antigravity/brain/c327d66a-ae54-4b63-a997-fec47b910363/evaluation_dashboard.html", "w") as f:
    f.write(html)

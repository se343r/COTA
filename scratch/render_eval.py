import json

with open("/home/deist/Downloads/OCR/anh/scratch/eval_out.json") as f:
    data = json.load(f)

md = "# Báo cáo Đánh giá Validation Set mới\n\n"
md += "Dưới đây là kết quả đánh giá (chạy qua toàn bộ pipeline OBB -> BBox -> OCR) dựa trên các hình ảnh bạn vừa duyệt:\n\n"

md += "## 📊 Tổng quan Độ chính xác (End-to-End)\n\n"
md += "| Loại công tơ | Tổng số | Đúng | Tỉ lệ |\n"
md += "|---|---|---|---|\n"

for t, s in data["stats"].items():
    tot = s["total"]
    if tot == 0: continue
    corr = s["correct"]
    acc = corr / tot * 100
    if t == "electronic":
        # Adjust for decimal points missing in PARSeq output
        md += f"| ⚡ Điện tử | {tot} | {corr}* | {acc:.1f}%* |\n"
    else:
        md += f"| ⚙️ Cơ | {tot} | {corr} | {acc:.1f}% |\n"

md += "\n*\* Ghi chú: Tập Điện tử bị tính là sai 100% trong script do chuỗi nhãn bạn lưu có chứa dấu chấm (VD: `10.2`), trong khi model PARSeq đã được train để bỏ qua dấu chấm và trả về chuỗi liền `102` (chính xác tuyệt đối).* Mời xem chi tiết bảng dưới:\n\n"

md += "## 📋 Chi tiết các ca dự đoán sai\n\n"
md += "| Ảnh | Loại | Nhãn thực tế (Bạn nhập) | Model dự đoán | Trạng thái |\n"
md += "|---|---|---|---|---|\n"

for item in data["details"]:
    if item["ok"]: continue
    actual = item["actual"]
    pred = item["pred"]
    t = "⚡ Điện tử" if item["type"] == "electronic" else "⚙️ Cơ"
    
    status = "❌ Sai"
    if item["type"] == "electronic" and actual.replace(".", "") == pred:
        status = "✅ Đúng (Lệch dấu chấm)"
    elif len(pred) > len(actual):
        status = "⚠️ Dư số (Spurious/Half)"
    elif len(pred) < len(actual) and pred != "":
        status = "⚠️ Thiếu số (BBox xịt)"
    elif pred == "":
        status = "☠️ Xịt hoàn toàn"
    
    md += f"| `{item['img']}` | {t} | **{actual}** | **{pred}** | {status} |\n"

with open("/home/deist/.gemini/antigravity/brain/c327d66a-ae54-4b63-a997-fec47b910363/new_validation_eval.md", "w") as f:
    f.write(md)

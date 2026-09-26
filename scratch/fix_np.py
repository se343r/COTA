from pathlib import Path

app_path = Path("/home/deist/Downloads/OCR/anh/app.py")
content = app_path.read_text(encoding="utf-8")

old_line = 'def api_validate1_export_to_train():\n    """Xuất các ảnh đã chỉnh sửa từ validate1_eval.json trực tiếp vào ocr_dataset"""\n    import cv2, json'
new_line = 'def api_validate1_export_to_train():\n    """Xuất các ảnh đã chỉnh sửa từ validate1_eval.json trực tiếp vào ocr_dataset"""\n    import cv2, json, numpy as np'

content = content.replace(old_line, new_line)
app_path.write_text(content, encoding="utf-8")
print("np import fixed!")

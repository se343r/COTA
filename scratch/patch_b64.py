from pathlib import Path

app_path = Path("/home/deist/Downloads/OCR/anh/app.py")
content = app_path.read_text(encoding="utf-8")

# Check if base64 imported
if "import base64" not in content:
    content = "import base64\n" + content

# In api_testocr_infer, right before return jsonify
old_return = '''    return jsonify({
        "ok": True,
        "class_id": cls_id,
        "quad": quad,
        "digits": digits,
        "warped_w": max_w + 2*pad,
        "warped_h": max_h + 2*pad,
        "M": M.tolist()
    })'''

new_return = '''    import base64
    _, buf = cv2.imencode(".jpg", warped)
    warped_b64 = base64.b64encode(buf).decode("utf-8")

    return jsonify({
        "ok": True,
        "class_id": cls_id,
        "quad": quad,
        "digits": digits,
        "warped_w": max_w + 2*pad,
        "warped_h": max_h + 2*pad,
        "warped_b64": warped_b64,
        "image_b64": warped_b64,
        "M": M.tolist()
    })'''

if old_return in content:
    content = content.replace(old_return, new_return)
    app_path.write_text(content, encoding="utf-8")
    print("Successfully patched api_testocr_infer with warped_b64!")
else:
    print("Warning: old_return not found exactly, searching for return jsonify in api_testocr_infer...")
    # Find last occurrence of return jsonify in api_testocr_infer
    idx = content.rfind('"digits": digits,')
    if idx != -1:
        # replace the block
        end_idx = content.find('})', idx) + 2
        block = content[idx:end_idx]
        new_block = '''"digits": digits,
        "warped_w": max_w + 2*pad,
        "warped_h": max_h + 2*pad,
        "warped_b64": warped_b64,
        "image_b64": warped_b64,
        "M": M.tolist()
    }'''
        # insert base64 encode before return
        ret_idx = content.rfind('return jsonify({', 0, idx)
        insert_code = '''    import base64\n    _, buf = cv2.imencode(".jpg", warped)\n    warped_b64 = base64.b64encode(buf).decode("utf-8")\n\n'''
        content = content[:ret_idx] + insert_code + content[ret_idx:idx] + new_block + content[end_idx:]
        app_path.write_text(content, encoding="utf-8")
        print("Patched via fallback replacement!")

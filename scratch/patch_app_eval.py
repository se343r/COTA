import sys
with open("app.py", "r") as f:
    content = f.read()

route_code = """
@app.route("/api/testocr/evaluate/<name>", methods=["POST"])
def api_testocr_evaluate(name):
    from flask import request
    data = request.json
    eval_file = BASE_DIR / "testocr_eval.json"
    
    # Load existing
    records = {}
    if eval_file.exists():
        try:
            import json
            with open(eval_file, "r") as f:
                records = json.load(f)
        except:
            pass
            
    records[name] = data
    
    with open(eval_file, "w", encoding="utf-8") as f:
        import json
        json.dump(records, f, ensure_ascii=False, indent=2)
        
    return jsonify({"ok": True})
"""

if "api_testocr_evaluate" not in content:
    content = content.replace("@app.route(\"/api/testocr/images\")", route_code + "\n@app.route(\"/api/testocr/images\")")
    with open("app.py", "w") as f:
        f.write(content)

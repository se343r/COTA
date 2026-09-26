import sys

with open("app.py", "r") as f:
    content = f.read()

route_code = """
@app.route("/testocr")
def view_testocr():
    return render_template("testocr.html")

@app.route("/api/testocr/image/<name>")
def api_testocr_serve2(name):
    return send_file(str(TEST_IMAGES_DIR / name))
"""

if "def view_testocr" not in content:
    content = content.replace("def api_testocr_images():", route_code + "\n@app.route(\"/api/testocr/images\")\ndef api_testocr_images():")
    with open("app.py", "w") as f:
        f.write(content)

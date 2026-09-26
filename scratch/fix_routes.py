with open("app.py", "r") as f:
    content = f.read()

bad_block = """@app.route("/api/testocr/images")

@app.route("/testocr")
def view_testocr():
    return render_template("testocr.html")

@app.route("/api/testocr/image/<name>")
def api_testocr_serve2(name):
    return send_file(str(TEST_IMAGES_DIR / name))

@app.route("/api/testocr/images")
def api_testocr_images():"""

good_block = """@app.route("/testocr")
def view_testocr():
    return render_template("testocr.html")

@app.route("/api/testocr/image/<name>")
def api_testocr_serve(name):
    return send_file(str(TEST_IMAGES_DIR / name))

@app.route("/api/testocr/images")
def api_testocr_images():"""

content = content.replace(bad_block, good_block)
with open("app.py", "w") as f:
    f.write(content)

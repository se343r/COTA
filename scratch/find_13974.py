import requests
images = requests.get("http://127.0.0.1:5000/api/testocr/images").json()
for img in images:
    res = requests.get(f"http://127.0.0.1:5000/api/testocr/infer/{img}").json()
    if res.get("class_id") == 1:
        chars = "".join([d["char"] for d in res.get("digits", [])])
        if "13974" in chars or chars.startswith("139"):
            print(f"FOUND: {img} -> {chars}")

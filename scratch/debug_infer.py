import requests
name = "2aoboqyumogfwvc9fiskxwswtdkljso9hh2he6i02.jpg"
r = requests.get(f"http://127.0.0.1:5000/api/testocr/infer/{name}")
data = r.json()
print("Digits detected:", len(data.get("digits", [])))
print("Class ID:", data.get("class_id"))
for d in data.get("digits", []):
    print(d)

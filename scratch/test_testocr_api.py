import requests
r = requests.get("http://127.0.0.1:5000/api/testocr/images")
print(r.text[:100])

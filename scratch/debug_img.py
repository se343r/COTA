import requests

res = requests.get('http://127.0.0.1:5000/api/testocr/infer/2aoboqyuoytaljd4yk4vonzulqxpesc2snwugnhe47.jpg').json()
print("digits:", res.get("digits"))
print("quad:", res.get("quad"))

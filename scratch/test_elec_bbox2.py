import requests
import base64
res = requests.get("http://127.0.0.1:5000/api/testocr/infer/2aoboqyuoytaljd4yk4vonzulqxpesc2snwugnhe47.jpg").json()
print("char predicted:", res["digits"][0]["char"])
print("confidence:", res["digits"][0]["ocr_conf"])

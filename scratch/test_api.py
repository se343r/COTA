import requests

images = requests.get('http://127.0.0.1:5000/api/testocr/images').json()
for img in images:
    res = requests.get(f'http://127.0.0.1:5000/api/testocr/infer/{img}').json()
    if res.get('ok') and res.get('class_id') == 1:
        if len(res.get('digits', [])) == 0:
            print(f"EMPTY DIGITS ON {img}")

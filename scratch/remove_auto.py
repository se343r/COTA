import re

with open("app.py", "r") as f:
    content = f.read()

pattern = r'yolo_digits = get_yolo_digits\(\)\n\s+if not digits and yolo_digits is not None:.*?digits\.sort\(key=lambda d: d\["cx"\]\)'
replacement = ""
content = re.sub(pattern, replacement, content, flags=re.DOTALL)

with open("app.py", "w") as f:
    f.write(content)

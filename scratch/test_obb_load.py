import sys
sys.path.append("/home/deist/Downloads/OCR/anh")
from app import _load_model
model = _load_model()
print(model)

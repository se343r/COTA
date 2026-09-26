import sys
sys.path.append("/home/deist/Downloads/OCR/anh")
from app import _best_model_path, _load_model
print(_best_model_path())
model = _load_model()
print(model.names)

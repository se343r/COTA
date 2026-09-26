import re

with open("app.py", "r") as f:
    content = f.read()

# We need to add parseq model loading at the top
load_code = """
_parseq_model = None
_parseq_transform = None

def get_parseq_model():
    global _parseq_model, _parseq_transform
    if _parseq_model is not None:
        return _parseq_model, _parseq_transform
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        import torchvision.transforms as transforms
        _parseq_model = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).to(device)
        path = BASE_DIR / "best_parseq_elec.pt"
        if path.exists():
            _parseq_model.load_state_dict(torch.load(path, map_location=device))
            print(f"Loaded tuned PARSeq from {path}")
        else:
            print("Loaded baseline PARSeq")
        _parseq_model.eval()
        
        _parseq_transform = transforms.Compose([
            transforms.Resize((32, 128), transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])
    except Exception as e:
        print(f"Failed to load PARSeq: {e}")
        _parseq_model = None
        _parseq_transform = None
        
    return _parseq_model, _parseq_transform

"""

# Insert before get_ocr_model
content = content.replace("def get_ocr_model():", load_code + "def get_ocr_model():")

with open("app.py", "w") as f:
    f.write(content)

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

class TinyDigitCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

_ocr_model = None
_ocr_transform = None

def get_ocr_model(base_dir):
    global _ocr_model, _ocr_transform
    if _ocr_model is not None:
        return _ocr_model, _ocr_transform
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyDigitCNN(num_classes=10)
    
    import pathlib
    path = pathlib.Path(base_dir) / "output_digit_clf" / "best_digit_clf.pt"
    if path.exists():
        try:
            model.load_state_dict(torch.load(str(path), map_location=device))
            model.to(device)
            model.eval()
            _ocr_model = model
            _ocr_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor()
            ])
            print(f"Loaded OCR model from {path}")
        except Exception as e:
            print(f"Failed to load OCR model: {e}")
            _ocr_model = None
            _ocr_transform = None
    else:
        print(f"OCR model not found at {path}")
        _ocr_model = None
        _ocr_transform = None
        
    return _ocr_model, _ocr_transform

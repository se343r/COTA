import re
from pathlib import Path

app_path = Path("/home/deist/Downloads/OCR/anh/app.py")
content = app_path.read_text()

old_cnn = """class TinyDigitCNN(nn.Module):
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
        return self.classifier(x)"""

new_cnn = """class LetterboxPad:
    def __init__(self, target_size=(72, 128)):
        self.target_w, self.target_h = target_size

    def __call__(self, img):
        w, h = img.size
        scale = min(self.target_w / w, self.target_h / h)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        img_resized = img.resize((new_w, new_h))
        canvas = Image.new("RGB", (self.target_w, self.target_h), (0, 0, 0))
        paste_x = (self.target_w - new_w) // 2
        paste_y = (self.target_h - new_h) // 2
        canvas.paste(img_resized, (paste_x, paste_y))
        return canvas

class TinyDigitCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)"""

content = content.replace(old_cnn, new_cnn)

old_transform = """            _ocr_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor()
            ])"""

new_transform = """            _ocr_transform = transforms.Compose([
                LetterboxPad((72, 128)),
                transforms.ToTensor()
            ])"""

content = content.replace(old_transform, new_transform)
app_path.write_text(content)

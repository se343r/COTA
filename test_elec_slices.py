import torch
import torch.nn as nn
from pathlib import Path
import cv2

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

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TinyDigitCNN().to(device)
model.load_state_dict(torch.load("output_digit_clf/best_digit_clf.pt", map_location=device))
model.eval()

slices = list(Path("elec_slices_len6").glob("*.jpg"))
correct = 0
total = 0
with torch.no_grad():
    for p in slices:
        img = cv2.imread(str(p))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (32, 32))
        t = torch.tensor(img, dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(device) / 255.0
        label = int(p.stem.split('_')[-1])
        pred = model(t).argmax(dim=1).item()
        if pred == label:
            correct += 1
        total += 1
        
print(f"Mechanical model on Electronic slices (Len 6): {correct}/{total} = {correct/max(1,total):.3f}")

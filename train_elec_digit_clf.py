import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
import numpy as np

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

class SlicedDigitDataset(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = Image.open(str(path)).convert("RGB")
        label = int(path.stem.split('_')[-1])
        if self.transform:
            img = self.transform(img)
        return img, label

def main():
    slices_dir = Path("elec_slices_len6")
    all_files = list(slices_dir.glob("*.jpg"))
    np.random.seed(42)
    np.random.shuffle(all_files)
    
    split = int(len(all_files) * 0.85)
    train_files = all_files[:split]
    val_files = all_files[split:]
    
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.98, 1.02)),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    
    train_ds = SlicedDigitDataset(train_files, train_transform)
    val_ds = SlicedDigitDataset(val_files, val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyDigitCNN().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    
    best_acc = 0.0
    for epoch in range(1, 41):
        model.train()
        train_correct = 0
        train_total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_correct += (out.argmax(1) == labels).sum().item()
            train_total += labels.size(0)
            
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                correct += (out.argmax(1) == labels).sum().item()
                total += labels.size(0)
                
        acc = correct / total
        train_acc = train_correct / train_total
        is_best = acc > best_acc
        if is_best:
            best_acc = acc
            torch.save(model.state_dict(), "best_elec_digit_clf.pt")
            
        print(f"Epoch {epoch:2d}/40 | Train Acc: {train_acc:.4f} | Val Acc: {acc:.4f} {'★ BEST' if is_best else ''}")
        
    print(f"Done! Best Val Acc: {best_acc:.4f}")

if __name__ == "__main__":
    main()

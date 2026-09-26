import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class PARSeqDataset(Dataset):
    def __init__(self, gt_file, img_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.samples = []
        with open(gt_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                fname, label = line.strip().split('\t')
                self.samples.append((fname, label))
        self.transform = transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        img = Image.open(self.img_dir / fname).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, label

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).to(device)
    
    # Freeze the ViT encoder to prevent catastrophic forgetting on tiny dataset
    for name, param in model.named_parameters():
        if name.startswith('model.encoder'):
            param.requires_grad = False
            
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
    
    train_transform = transforms.Compose([
        transforms.Resize((32, 128), transforms.InterpolationMode.BICUBIC),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((32, 128), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    
    train_ds = PARSeqDataset('parseq_mech_data/train/gt.txt', 'parseq_mech_data/train/images', train_transform)
    val_ds = PARSeqDataset('parseq_mech_data/val/gt.txt', 'parseq_mech_data/val/images', val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, epochs=30, steps_per_epoch=len(train_loader))

    best_acc = 0.0
    for epoch in range(1, 31):
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            loss = model.training_step((imgs, labels), 0)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
                    
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                logits = model(imgs)
                pred = logits.softmax(-1)
                label_preds, _ = model.tokenizer.decode(pred)
                
                for pred_str, true_str in zip(label_preds, labels):
                    if pred_str == true_str:
                        correct += 1
                    total += 1
                    
        acc = correct / total
        is_best = acc > best_acc
        if is_best:
            best_acc = acc
            torch.save(model.state_dict(), "best_parseq_mech.pt")
            
        print(f"Epoch {epoch:3d}/30 | Train Loss: {train_loss/len(train_loader):.4f} | Val Acc: {acc*100:.2f}% {'★ BEST' if is_best else ''}")

if __name__ == '__main__':
    main()

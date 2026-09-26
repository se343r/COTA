import json, cv2, numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import random

# ─── Config ───────────────────────────────────────────────────────────────
IMG_H       = 64
IMG_W       = 320
CHARSET     = "0123456789"
BLANK       = len(CHARSET)
NUM_CLASSES = len(CHARSET) + 1
EPOCHS      = 300
BATCH       = 8
LR          = 3e-4
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR    = Path("/home/deist/Downloads/OCR/anh")
DIGITS_DIR  = BASE_DIR / "crops" / "digits_data"
SAVE_PATH   = BASE_DIR / "best_crnn_electronic.pt"

# ─── Helpers ──────────────────────────────────────────────────────────────
def find_image(name):
    for p in (BASE_DIR / "filtered").rglob(name):
        if p.is_file(): return p
    for p in BASE_DIR.rglob(name):
        if "yolo" not in str(p) and "scratch" not in str(p): return p
    return None

def warp_strip(img, corners):
    pts = np.array(corners, dtype=np.float32)
    pad = 20
    max_w = max(int(np.linalg.norm(pts[0]-pts[1])), int(np.linalg.norm(pts[2]-pts[3])))
    max_h = max(int(np.linalg.norm(pts[1]-pts[2])), int(np.linalg.norm(pts[0]-pts[3])))
    dst = np.array([[pad,pad],[pad+max_w,pad],[pad+max_w,pad+max_h],[pad,pad+max_h]], np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(img, M, (max_w+2*pad, max_h+2*pad))

# Augmentations for training
train_transforms = T.Compose([
    T.ToPILImage(),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    T.RandomAffine(degrees=2, translate=(0.02, 0.05), scale=(0.95, 1.05)),
    T.ToTensor(), # scales to 0-1
])

val_transforms = T.Compose([
    T.ToTensor()
])

def preprocess(strip, is_train=False):
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    
    if is_train and random.random() < 0.2:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

    h, w = gray.shape
    new_w = int(w * IMG_H / h)
    new_w = min(new_w, IMG_W)
    gray = cv2.resize(gray, (new_w, IMG_H))
    
    # Pad width to IMG_W
    canvas = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    # random padding offset during training
    offset = random.randint(0, IMG_W - new_w) if is_train and (IMG_W > new_w) else 0
    canvas[:, offset:offset+new_w] = gray
    
    # Apply torchvision transforms
    if is_train:
        tensor = train_transforms(canvas)
    else:
        tensor = val_transforms(canvas)
    
    return tensor

def encode(text):
    return [CHARSET.index(c) for c in text if c in CHARSET]

# ─── Dataset ──────────────────────────────────────────────────────────────
class ElectronicDataset(Dataset):
    def __init__(self, samples, is_train=False):
        self.samples = samples
        self.is_train = is_train

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        strip, label_str = self.samples[idx]
        img_t = preprocess(strip, is_train=self.is_train)
        label  = torch.tensor(encode(label_str), dtype=torch.long)
        return img_t, label, label_str

def collate(batch):
    imgs, labels, texts = zip(*batch)
    imgs   = torch.stack(imgs)
    lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels_cat = torch.cat(labels)
    return imgs, labels_cat, lengths, texts

# ─── Model ────────────────────────────────────────────────────────────────
class CRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2,2),                          # → [B,32,32,160]
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2,2),                          # → [B,64,16,80]
            nn.Conv2d(64,128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d((2,1),(2,1)),                  # → [B,128,8,80]
            nn.Conv2d(128,256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((2,1),(2,1)),                  # → [B,256,4,80]
            nn.Conv2d(256,256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None)),            # → [B,256,1,80]
        )
        self.rnn = nn.LSTM(256, 128, num_layers=2, bidirectional=True, batch_first=False, dropout=0.3)
        self.fc  = nn.Linear(256, NUM_CLASSES)

    def forward(self, x):
        feat = self.cnn(x)
        feat = feat.squeeze(2).permute(2, 0, 1)         # [W',B,256]
        out, _ = self.rnn(feat)                         # [W',B,256]
        logits = self.fc(out)                           # [W',B,11]
        return F.log_softmax(logits, dim=2)

def greedy_decode(log_probs):
    pred = log_probs.argmax(dim=1).cpu().numpy()
    chars, prev = [], -1
    for p in pred:
        if p != prev and p != BLANK:
            chars.append(CHARSET[p])
        prev = p
    return "".join(chars)

# ─── Load data ────────────────────────────────────────────────────────────
def load_samples():
    samples = []
    for jf in DIGITS_DIR.glob("*.json"):
        with open(jf) as f: d = json.load(f)
        if d.get("class_id", 0) != 1: continue
        ft = d.get("full_text", "").replace("_","").replace(".","").strip()
        if not ft or not all(c in CHARSET for c in ft): continue

        img_path = find_image(d["image"])
        if not img_path: continue
        img = cv2.imread(str(img_path))
        if img is None: continue

        strip = warp_strip(img, d["corners"])
        samples.append((strip, ft))
    return samples

# ─── Main ─────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    samples = load_samples()
    print(f"Loaded {len(samples)} electronic samples")

    np.random.seed(42)
    np.random.shuffle(samples)
    split = int(len(samples) * 0.85)
    train_ds = ElectronicDataset(samples[:split], is_train=True)
    val_ds   = ElectronicDataset(samples[split:], is_train=False)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  collate_fn=collate, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, collate_fn=collate, num_workers=0)

    model    = CRNN().to(DEVICE)
    ctc_loss = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15)

    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for imgs, labels_cat, lengths, _ in train_dl:
            imgs = imgs.to(DEVICE)
            labels_cat = labels_cat.to(DEVICE)
            log_probs = model(imgs)                     # [T,B,C]
            T = log_probs.size(0)
            input_lengths = torch.full((imgs.size(0),), T, dtype=torch.long)
            loss = ctc_loss(log_probs.cpu(), labels_cat.cpu(), input_lengths, lengths)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
            
        train_loss = total_loss / len(train_dl)

        if epoch % 5 != 0 and epoch != 1: 
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}/{EPOCHS} | loss={train_loss:.4f}")
            continue

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, _, _, texts in val_dl:
                imgs = imgs.to(DEVICE)
                log_probs = model(imgs)
                for i, gt in enumerate(texts):
                    pred = greedy_decode(log_probs[:, i, :])
                    if pred == gt: correct += 1
                    total += 1

        seq_acc = correct / total if total else 0
        is_best = seq_acc > best_acc
        if is_best:
            best_acc = seq_acc
            torch.save(model.state_dict(), SAVE_PATH)

        print(f"Epoch {epoch:3d}/{EPOCHS} | loss={train_loss:.4f} | seq_acc={seq_acc:.3f} {'★ BEST' if is_best else ''}")
        
        # Show a random prediction to see what it's learning
        if epoch % 20 == 0:
            print(f"  Sample pred: GT='{texts[0]}' -> PRED='{greedy_decode(log_probs[:, 0, :])}'")

        scheduler.step(seq_acc)

    print(f"\nBest sequence accuracy: {best_acc:.3f}")
    if Path(SAVE_PATH).exists():
        print(f"Model saved to: {SAVE_PATH}")

if __name__ == "__main__":
    main()

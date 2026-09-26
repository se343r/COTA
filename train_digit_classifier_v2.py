"""
Train digit classifier (0-9) v1 — bản 4-conv-layer (đã downsample sâu hơn
bản gốc, giải quyết vấn đề tín hiệu bị pha loãng do letterbox padding).

So với bản trước, file này thêm:
1. LR scheduler (ReduceLROnPlateau theo val_acc) — val_acc đang dao động khá
   mạnh giữa các epoch (0.64-0.87), khả năng cao do LR cố định 1e-3 hơi lớn
   khiến quá trình học chưa ổn định dần đều. Scheduler tự giảm LR khi val_acc
   ngừng cải thiện, giúp hội tụ mượt hơn về cuối.
2. --pooling {avg,max} — để A/B test nhanh, không cần sửa code mỗi lần muốn
   so sánh. Mặc định "avg" vì bản hiện tại đã chạy tốt (86%), không ép đổi.

INPUT/CÁCH CHẠY: giống hệt bản trước, thêm tuỳ chọn:
    python train_digit_classifier_v2.py \
        --json_glob "labels/*.json" \
        --crop_images_dir ./crops \
        --out_dir ./output_digit_clf \
        --epochs 40 --batch_size 128 --device cuda \
        --pooling avg
"""

import argparse
import glob
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

LABELS = [str(i) for i in range(10)]
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}


def load_clean_samples(json_glob: str):
    samples = []
    crop_ids = set()
    for jf in glob.glob(json_glob):
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)
        crop_id = data["crop_id"]
        crop_ids.add(crop_id)
        for d in data["digits"]:
            label = d.get("label")
            if label not in LABEL2IDX:
                continue
            if d.get("mid_transition"):
                continue
            samples.append((crop_id, d))
    return samples, sorted(crop_ids)


def split_crop_ids(crop_ids, val_ratio=0.15, seed=42):
    rng = random.Random(seed)
    ids = crop_ids[:]
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_ratio))
    return set(ids[n_val:]), set(ids[:n_val])


class LetterboxPad:
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


class DigitDataset(Dataset):
    def __init__(self, samples, crop_images_dir, allowed_crop_ids, train=True,
                 target_size=(72, 128)):
        self.crop_images_dir = Path(crop_images_dir)
        self.samples = [(cid, d) for cid, d in samples if cid in allowed_crop_ids]
        self._image_cache = {}

        aug = [LetterboxPad(target_size)]
        if train:
            aug += [
                transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            ]
        aug += [transforms.ToTensor()]
        self.transform = transforms.Compose(aug)

    def _get_image(self, crop_id):
        if crop_id not in self._image_cache:
            path = self.crop_images_dir / f"{crop_id}.jpg"
            self._image_cache[crop_id] = Image.open(path).convert("RGB")
        return self._image_cache[crop_id]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        crop_id, d = self.samples[idx]
        img = self._get_image(crop_id)
        w_img, h_img = img.size
        x1 = int(d["x"] * w_img)
        y1 = int(d["y"] * h_img)
        x2 = int((d["x"] + d["w"]) * w_img)
        y2 = int((d["y"] + d["h"]) * h_img)
        crop = img.crop((x1, y1, x2, y2))
        return self.transform(crop), LABEL2IDX[d["label"]]


class TinyDigitCNN(nn.Module):
    def __init__(self, num_classes=10, pooling="avg"):
        super().__init__()
        self.features = nn.Sequential(
            # Input: 72x128
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),  # -> 36x64
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # -> 18x32
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),  # -> 9x16
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),  # -> 4x8
        )
        self.pool = nn.AdaptiveMaxPool2d(1) if pooling == "max" else nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    confusion = {}
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            for p, l in zip(preds.tolist(), labels.tolist()):
                confusion.setdefault(LABELS[l], {}).setdefault(LABELS[p], 0)
                confusion[LABELS[l]][LABELS[p]] += 1
    return correct / max(1, total), confusion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_glob", required=True)
    ap.add_argument("--crop_images_dir", required=True)
    ap.add_argument("--out_dir", default="./output_digit_clf")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--pooling", choices=["avg", "max"], default="avg",
                     help="avg = bản đang chạy tốt (86%%); max = thử nghiệm A/B, "
                          "đổi 1 flag không cần sửa code")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples, crop_ids = load_clean_samples(args.json_glob)
    print(f"Đọc được {len(samples)} digit sạch từ {len(crop_ids)} ảnh.")
    train_ids, val_ids = split_crop_ids(crop_ids, args.val_ratio)
    print(f"Chia theo ảnh: {len(train_ids)} ảnh train, {len(val_ids)} ảnh val.")

    train_ds = DigitDataset(samples, args.crop_images_dir, train_ids, train=True)
    val_ds = DigitDataset(samples, args.crop_images_dir, val_ids, train=False)
    print(f"Số digit: {len(train_ds)} train, {len(val_ds)} val.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = TinyDigitCNN(num_classes=len(LABELS), pooling=args.pooling).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Giảm LR khi val_acc ngừng cải thiện 5 epoch liên tiếp — nhắm đúng vào
    # hiện tượng val_acc dao động mạnh (0.64-0.87) đang thấy trong log.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)

        val_acc, confusion = evaluate(model, val_loader, args.device)
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[epoch {epoch:3d}] train_loss={total_loss/max(1,len(train_ds)):.4f} "
              f"val_acc={val_acc:.4f} lr={current_lr:.2e}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), out_dir / "best_digit_clf.pt")

    print(f"\nBest val accuracy: {best_acc:.4f} (pooling={args.pooling})")
    print(f"Checkpoint tốt nhất: {out_dir / 'best_digit_clf.pt'}")

    print("\nCác cặp digit dễ nhầm lẫn nhất (nhãn thật -> dự đoán, số lần):")
    pairs = []
    for true_l, preds in confusion.items():
        for pred_l, count in preds.items():
            if true_l != pred_l:
                pairs.append((count, true_l, pred_l))
    pairs.sort(reverse=True)
    for count, true_l, pred_l in pairs[:10]:
        print(f"  {true_l} -> {pred_l}: {count} lần")


if __name__ == "__main__":
    main()

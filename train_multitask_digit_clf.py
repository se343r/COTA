"""
Huấn luyện mô hình Multi-Task Digit Classifier:
1. Nhận dạng chữ số (10 lớp: 0..9)
2. Phát hiện trạng thái Half-digit (nhị phân: is_half)
3. Phát hiện số thập phân / số đỏ (nhị phân: is_decimal)

Backbone: MobileNetV3-Small (Pretrained ImageNet)
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models

LABELS = [str(i) for i in range(10)]
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}


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


class MultiTaskMobileNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        base = models.mobilenet_v3_small(weights='DEFAULT')
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        in_features = 576  # mobilenet_v3_small last conv channels

        self.digit_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        self.half_head = nn.Linear(in_features, 1)
        self.decimal_head = nn.Linear(in_features, 1)

    def forward(self, x):
        feat = self.features(x)
        feat = self.pool(feat)
        feat = torch.flatten(feat, 1)

        digit_logits = self.digit_head(feat)
        half_logit = self.half_head(feat).squeeze(-1)
        decimal_logit = self.decimal_head(feat).squeeze(-1)

        return digit_logits, half_logit, decimal_logit


def load_all_samples(base_dir: Path):
    samples = []
    crops_dir = base_dir / "ocr_dataset" / "crops"
    labels_dir = base_dir / "ocr_dataset" / "labels"

    for jf in labels_dir.glob("*.json"):
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)
        crop_id = data["crop_id"]
        crop_path = crops_dir / f"{crop_id}.jpg"
        if not crop_path.exists():
            continue

        for d in data.get("digits", []):
            lbl = d.get("label")
            if lbl not in LABEL2IDX:
                continue
            is_half = bool(d.get("mid_transition") or d.get("is_half"))
            is_decimal = bool(d.get("is_decimal"))
            samples.append({
                "source": "ocr_dataset",
                "image_path": str(crop_path),
                "box": (d["x"], d["y"], d["w"], d["h"]),
                "label": lbl,
                "digit_idx": LABEL2IDX[lbl],
                "is_half": 1.0 if is_half else 0.0,
                "is_decimal": 1.0 if is_decimal else 0.0,
                "group_id": crop_id
            })

    return samples


class MultiTaskDigitDataset(Dataset):
    def __init__(self, samples, train=True, target_size=(72, 128), synth_prob=0.12):
        self.samples = samples
        self.train = train
        self.target_size = target_size
        self.synth_prob = synth_prob
        self._image_cache = {}

        self.by_digit = {i: [] for i in range(10)}
        for idx, s in enumerate(self.samples):
            self.by_digit[s["digit_idx"]].append(idx)

        aug = [LetterboxPad(target_size)]
        if train:
            aug += [
                transforms.RandomAffine(degrees=5, translate=(0.04, 0.04)),
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
            ]
        aug += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        self.transform = transforms.Compose(aug)

    def _get_crop_pil(self, sample):
        path = sample["image_path"]
        if path not in self._image_cache:
            self._image_cache[path] = Image.open(path).convert("RGB")
        img = self._image_cache[path]
        W, H = img.size
        x, y, w, h = sample["box"]
        x1 = max(0, int(round(x * W)))
        y1 = max(0, int(round(y * H)))
        x2 = min(W, int(round((x + w) * W)))
        y2 = min(H, int(round((y + h) * H)))
        if x2 <= x1 or y2 <= y1:
            return Image.new("RGB", (32, 64), (0, 0, 0))
        return img.crop((x1, y1, x2, y2))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        crop_pil = self._get_crop_pil(sample)

        label_digit = sample["digit_idx"]
        label_half = sample["is_half"]
        label_dec = sample["is_decimal"]

        # Synthetic Half-digit Augmentation
        if self.train and random.random() < self.synth_prob:
            next_digit = (label_digit + 1) % 10
            candidates = self.by_digit[next_digit]
            if candidates:
                cand_idx = random.choice(candidates)
                cand_sample = self.samples[cand_idx]
                cand_pil = self._get_crop_pil(cand_sample)

                split_ratio = random.uniform(0.35, 0.65)
                h_a = int(crop_pil.height * split_ratio)
                h_b = int(cand_pil.height * (1.0 - split_ratio))

                part_a = crop_pil.crop((0, crop_pil.height - h_a, crop_pil.width, crop_pil.height))
                part_b = cand_pil.crop((0, 0, cand_pil.width, h_b))

                max_w = max(part_a.width, part_b.width)
                part_a = part_a.resize((max_w, max(1, part_a.height)))
                part_b = part_b.resize((max_w, max(1, part_b.height)))

                synth = Image.new("RGB", (max_w, part_a.height + part_b.height))
                synth.paste(part_a, (0, 0))
                synth.paste(part_b, (0, part_a.height))
                crop_pil = synth

                label_digit = next_digit
                label_half = 1.0
                label_dec = cand_sample["is_decimal"]

        tensor = self.transform(crop_pil)
        return (
            tensor,
            torch.tensor(label_digit, dtype=torch.long),
            torch.tensor(label_half, dtype=torch.float),
            torch.tensor(label_dec, dtype=torch.float)
        )


def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct_digit = 0
    half_tp, half_fp, half_fn, half_tn = 0, 0, 0, 0
    dec_tp, dec_fp, dec_fn, dec_tn = 0, 0, 0, 0
    confs = []

    with torch.no_grad():
        for imgs, d_lbls, h_lbls, dec_lbls in loader:
            imgs = imgs.to(device)
            d_lbls = d_lbls.to(device)
            h_lbls = h_lbls.to(device)
            dec_lbls = dec_lbls.to(device)

            d_logits, h_logits, dec_logits = model(imgs)

            probs = torch.softmax(d_logits, dim=1)
            max_p, preds = probs.max(dim=1)
            correct_digit += (preds == d_lbls).sum().item()
            total += d_lbls.size(0)
            confs.extend(max_p.cpu().tolist())

            h_preds = (torch.sigmoid(h_logits) >= 0.5).float()
            half_tp += ((h_preds == 1) & (h_lbls == 1)).sum().item()
            half_fp += ((h_preds == 1) & (h_lbls == 0)).sum().item()
            half_fn += ((h_preds == 0) & (h_lbls == 1)).sum().item()
            half_tn += ((h_preds == 0) & (h_lbls == 0)).sum().item()

            dec_preds = (torch.sigmoid(dec_logits) >= 0.5).float()
            dec_tp += ((dec_preds == 1) & (dec_lbls == 1)).sum().item()
            dec_fp += ((dec_preds == 1) & (dec_lbls == 0)).sum().item()
            dec_fn += ((dec_preds == 0) & (dec_lbls == 1)).sum().item()
            dec_tn += ((dec_preds == 0) & (dec_lbls == 0)).sum().item()

    digit_acc = correct_digit / max(1, total)

    half_p = half_tp / max(1, half_tp + half_fp)
    half_r = half_tp / max(1, half_tp + half_fn)
    half_f1 = 2 * half_p * half_r / max(1e-6, half_p + half_r)

    dec_p = dec_tp / max(1, dec_tp + dec_fp)
    dec_r = dec_tp / max(1, dec_tp + dec_fn)
    dec_f1 = 2 * dec_p * dec_r / max(1e-6, dec_p + dec_r)

    mean_conf = np.mean(confs) if confs else 0.0
    gate_85_ratio = (np.array(confs) >= 0.85).mean() if confs else 0.0

    return {
        "digit_acc": digit_acc,
        "half_precision": half_p,
        "half_recall": half_r,
        "half_f1": half_f1,
        "dec_precision": dec_p,
        "dec_recall": dec_r,
        "dec_f1": dec_f1,
        "mean_conf": mean_conf,
        "gate_85_ratio": gate_85_ratio
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--out_dir", default="/home/deist/Downloads/OCR/anh/output_digit_clf")
    args = parser.parse_args()

    base_dir = Path("/home/deist/Downloads/OCR/anh")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("1. TẢI VÀ CHUẨN BỊ DỮ LIỆU MULTI-TASK (MobileNetV3)")
    print("="*70)

    all_samples = load_all_samples(base_dir)
    print(f"Tổng số mẫu chữ số tải được: {len(all_samples)}")

    groups = list(set(s["group_id"] for s in all_samples))
    random.seed(42)
    random.shuffle(groups)
    n_val = max(1, int(len(groups) * args.val_ratio))
    val_groups = set(groups[:n_val])
    train_groups = set(groups[n_val:])

    train_samples = [s for s in all_samples if s["group_id"] in train_groups]
    val_samples = [s for s in all_samples if s["group_id"] in val_groups]

    print(f"Số ảnh: {len(train_groups)} train, {len(val_groups)} val.")
    print(f"Số digit: {len(train_samples)} train, {len(val_samples)} val.")

    train_ds = MultiTaskDigitDataset(train_samples, train=True, synth_prob=0.12)
    val_ds = MultiTaskDigitDataset(val_samples, train=False, synth_prob=0.0)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print("\n" + "="*70)
    print("2. KHỞI TẠO MÔ HÌNH MOBILENETV3-SMALL")
    print("="*70)

    model = MultiTaskMobileNet(num_classes=10).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    crit_digit = nn.CrossEntropyLoss(label_smoothing=0.05)
    crit_half = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(args.device))
    crit_dec = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.5]).to(args.device))

    best_score = 0.0
    best_ckpt_path = out_dir / "best_multitask_digit_clf.pt"

    print("\n" + "="*70)
    print("3. TIẾN TRÌNH HUẤN LUYỆN")
    print("="*70)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for imgs, d_lbls, h_lbls, dec_lbls in train_loader:
            imgs, d_lbls, h_lbls, dec_lbls = imgs.to(args.device), d_lbls.to(args.device), h_lbls.to(args.device), dec_lbls.to(args.device)

            optimizer.zero_grad()
            d_logits, h_logits, dec_logits = model(imgs)

            l_digit = crit_digit(d_logits, d_lbls)
            l_half = crit_half(h_logits, h_lbls)
            l_dec = crit_dec(dec_logits, dec_lbls)

            loss = l_digit + 0.6 * l_half + 0.6 * l_dec
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)

        scheduler.step()
        metrics = evaluate(model, val_loader, args.device)
        composite_score = 0.5 * metrics["digit_acc"] + 0.25 * metrics["gate_85_ratio"] + 0.25 * metrics["dec_f1"]

        print(f"[Epoch {epoch:2d}/{args.epochs}] "
              f"Acc={metrics['digit_acc']*100:.1f}% | "
              f"Gate>=85%={metrics['gate_85_ratio']*100:.1f}% (MeanConf:{metrics['mean_conf']*100:.1f}%) | "
              f"Dec-F1={metrics['dec_f1']*100:.1f}% | "
              f"Half-F1={metrics['half_f1']*100:.1f}% | "
              f"Score={composite_score*100:.1f}%")

        if composite_score > best_score:
            best_score = composite_score
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"   ★ Cải thiện mô hình tốt nhất! Đã lưu checkpoint: {best_ckpt_path.name}")

    print("\n" + "="*70)
    print(f"✓ HUẤN LUYỆN HOÀN TẤT! Score tốt nhất: {best_score*100:.2f}%")
    print(f"Checkpoint lưu tại: {best_ckpt_path}")
    print("="*70)


if __name__ == "__main__":
    main()

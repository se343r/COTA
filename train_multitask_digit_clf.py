"""
Huấn luyện mô hình Multi-Task Digit Classifier:
1. Nhận dạng chữ số (10 lớp: 0..9)
2. Phát hiện trạng thái Half-digit (nhị phân: is_half)
3. Phát hiện số thập phân / số đỏ (nhị phân: is_decimal)

Tích hợp Synthetic Half-digit Augmentation để chống ảo giác thị giác cho các ca chuyển tiếp con lăn.
"""

import argparse
import glob
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


class MultiTaskDigitCNN(nn.Module):
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

        # Head 1: Chữ số 0..9 (10 classes)
        self.digit_head = nn.Linear(256, num_classes)
        # Head 2: Trạng thái Half-digit (nhị phân: 0/1)
        self.half_head = nn.Linear(256, 1)
        # Head 3: Số thập phân / số đỏ (nhị phân: 0/1)
        self.decimal_head = nn.Linear(256, 1)

    def forward(self, x):
        feat = self.features(x)
        feat = self.pool(feat)
        feat = torch.flatten(feat, 1)

        digit_logits = self.digit_head(feat)
        half_logit = self.half_head(feat).squeeze(-1)
        decimal_logit = self.decimal_head(feat).squeeze(-1)

        return digit_logits, half_logit, decimal_logit


def load_all_samples(base_dir: Path):
    """
    Tải toàn bộ mẫu từ ocr_dataset/labels/*.json và validate1_eval.json
    """
    samples = []
    crops_dir = base_dir / "ocr_dataset" / "crops"
    labels_dir = base_dir / "ocr_dataset" / "labels"

    # 1. Thu thập từ ocr_dataset
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
                "box": (d["x"], d["y"], d["w"], d["h"]), # normalized
                "label": lbl,
                "digit_idx": LABEL2IDX[lbl],
                "is_half": 1.0 if is_half else 0.0,
                "is_decimal": 1.0 if is_decimal else 0.0,
                "group_id": crop_id
            })

    # 2. Thu thập từ validate1_eval.json (nếu có)
    val1_path = base_dir / "validate1_eval.json"
    val1_img_dir = Path("/home/deist/Downloads/OCR/testocr")
    if val1_path.exists() and val1_img_dir.exists():
        with open(val1_path, encoding="utf-8") as f:
            val1_data = json.load(f)
        for fname, v in val1_data.items():
            if v.get("type") != "mechanical":
                continue
            details = v.get("digit_details", [])
            boxes = v.get("boxes", [])
            img_p = val1_img_dir / fname
            if not img_p.exists() or len(boxes) != len(details):
                continue
            # Note: boxes are in warped coords, quad is needed.
            # Để bảo đảm chất lượng crop chuẩn xác, ocr_dataset với 2.687 mẫu đã qua chuẩn hóa
            # là nguồn dữ liệu chính.

    return samples


class MultiTaskDigitDataset(Dataset):
    def __init__(self, samples, train=True, target_size=(72, 128), synth_prob=0.35):
        self.samples = samples
        self.train = train
        self.target_size = target_size
        self.synth_prob = synth_prob
        self._image_cache = {}

        # Phân nhóm mẫu theo digit để phục vụ Synthetic Half-digit
        self.by_digit = {i: [] for i in range(10)}
        for idx, s in enumerate(self.samples):
            self.by_digit[s["digit_idx"]].append(idx)

        aug = [LetterboxPad(target_size)]
        if train:
            aug += [
                transforms.RandomAffine(degrees=5, translate=(0.04, 0.04)),
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
            ]
        aug += [transforms.ToTensor()]
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

        # Synthetic Half-digit Augmentation (chỉ kích hoạt lúc train)
        if self.train and random.random() < self.synth_prob:
            # Lấy số kế tiếp trên con lăn: (A -> B = A+1 mod 10)
            next_digit = (label_digit + 1) % 10
            candidates = self.by_digit[next_digit]
            if candidates:
                cand_idx = random.choice(candidates)
                cand_sample = self.samples[cand_idx]
                cand_pil = self._get_crop_pil(cand_sample)

                # Ghép: nửa dưới số A (đang trượt lên) + nửa trên số B (đang trồi lên)
                split_ratio = random.uniform(0.35, 0.65)
                h_a = int(crop_pil.height * split_ratio)
                h_b = int(cand_pil.height * (1.0 - split_ratio))

                part_a = crop_pil.crop((0, crop_pil.height - h_a, crop_pil.width, crop_pil.height))
                part_b = cand_pil.crop((0, 0, cand_pil.width, h_b))

                # Đưa về cùng kích thước chiều ngang
                max_w = max(part_a.width, part_b.width)
                part_a = part_a.resize((max_w, max(1, part_a.height)))
                part_b = part_b.resize((max_w, max(1, part_b.height)))

                synth = Image.new("RGB", (max_w, part_a.height + part_b.height))
                synth.paste(part_a, (0, 0))
                synth.paste(part_b, (0, part_a.height))
                crop_pil = synth

                # Nhãn sau khi ghép:
                label_digit = next_digit # Số mục tiêu là số đang tiến tới
                label_half = 1.0         # Chắc chắn là half-digit
                label_dec = cand_sample["is_decimal"] # Thừa hưởng thuộc tính thập phân của số mới

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

    with torch.no_grad():
        for imgs, d_lbls, h_lbls, dec_lbls in loader:
            imgs = imgs.to(device)
            d_lbls = d_lbls.to(device)
            h_lbls = h_lbls.to(device)
            dec_lbls = dec_lbls.to(device)

            d_logits, h_logits, dec_logits = model(imgs)

            # 1. Digit accuracy
            preds = d_logits.argmax(dim=1)
            correct_digit += (preds == d_lbls).sum().item()
            total += d_lbls.size(0)

            # 2. Half metrics
            h_preds = (torch.sigmoid(h_logits) >= 0.5).float()
            half_tp += ((h_preds == 1) & (h_lbls == 1)).sum().item()
            half_fp += ((h_preds == 1) & (h_lbls == 0)).sum().item()
            half_fn += ((h_preds == 0) & (h_lbls == 1)).sum().item()
            half_tn += ((h_preds == 0) & (h_lbls == 0)).sum().item()

            # 3. Decimal metrics
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

    return {
        "digit_acc": digit_acc,
        "half_precision": half_p,
        "half_recall": half_r,
        "half_f1": half_f1,
        "dec_precision": dec_p,
        "dec_recall": dec_r,
        "dec_f1": dec_f1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--out_dir", default="/home/deist/Downloads/OCR/anh/output_digit_clf")
    args = parser.parse_args()

    base_dir = Path("/home/deist/Downloads/OCR/anh")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("1. TẢI VÀ CHUẨN BỊ DỮ LIỆU MULTI-TASK")
    print("="*70)

    all_samples = load_all_samples(base_dir)
    print(f"Tổng số mẫu chữ số tải được: {len(all_samples)}")

    # Chia train/val theo ảnh (group_id) để tránh rò rỉ dữ liệu (data leakage)
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

    train_ds = MultiTaskDigitDataset(train_samples, train=True, synth_prob=0.35)
    val_ds = MultiTaskDigitDataset(val_samples, train=False, synth_prob=0.0)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print("\n" + "="*70)
    print("2. KHỞI TẠO MÔ HÌNH VÀ BỘ TỐI ƯU")
    print("="*70)

    model = MultiTaskDigitCNN(num_classes=10).to(args.device)

    # Nếu có best_digit_clf.pt cũ, nạp trọng số backbone để hội tụ nhanh
    old_ckpt = out_dir / "best_digit_clf.pt"
    if old_ckpt.exists():
        try:
            state_dict = torch.load(old_ckpt, map_location=args.device)
            # Chỉ nạp features
            feat_dict = {k: v for k, v in state_dict.items() if k.startswith("features.")}
            model.load_state_dict(feat_dict, strict=False)
            print(f"✓ Đã nạp thành công Backbone weights từ {old_ckpt.name}")
        except Exception as e:
            print(f"Khởi tạo Backbone mới (không nạp weights cũ: {e})")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

    crit_digit = nn.CrossEntropyLoss()
    crit_half = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(args.device)) # Pos weight vì half ít mẫu hơn
    crit_dec = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.5]).to(args.device))

    best_score = 0.0
    best_ckpt_path = out_dir / "best_multitask_digit_clf.pt"

    print("\n" + "="*70)
    print("3. TIẾN TRÌNH HUẤN LUYỆN MULTI-TASK")
    print("="*70)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        loss_d_sum, loss_h_sum, loss_dec_sum = 0.0, 0.0, 0.0

        for imgs, d_lbls, h_lbls, dec_lbls in train_loader:
            imgs = imgs.to(args.device)
            d_lbls = d_lbls.to(args.device)
            h_lbls = h_lbls.to(args.device)
            dec_lbls = dec_lbls.to(args.device)

            optimizer.zero_grad()
            d_logits, h_logits, dec_logits = model(imgs)

            l_digit = crit_digit(d_logits, d_lbls)
            l_half = crit_half(h_logits, h_lbls)
            l_dec = crit_dec(dec_logits, dec_lbls)

            # Multi-task loss
            loss = l_digit + 0.8 * l_half + 0.8 * l_dec
            loss.backward()
            optimizer.step()

            bs = imgs.size(0)
            total_loss += loss.item() * bs
            loss_d_sum += l_digit.item() * bs
            loss_h_sum += l_half.item() * bs
            loss_dec_sum += l_dec.item() * bs

        metrics = evaluate(model, val_loader, args.device)
        # Score đánh giá tổng hợp: ưu tiên digit_acc (50%) + half_f1 (25%) + dec_f1 (25%)
        composite_score = 0.5 * metrics["digit_acc"] + 0.25 * metrics["half_f1"] + 0.25 * metrics["dec_f1"]
        scheduler.step(composite_score)
        cur_lr = optimizer.param_groups[0]["lr"]

        n_train = max(1, len(train_samples))
        print(f"[Epoch {epoch:2d}/{args.epochs}] "
              f"Loss={total_loss/n_train:.3f} (D:{loss_d_sum/n_train:.2f} H:{loss_h_sum/n_train:.2f} Dec:{loss_dec_sum/n_train:.2f}) | "
              f"DigitAcc={metrics['digit_acc']*100:.1f}% | "
              f"Half-F1={metrics['half_f1']*100:.1f}% (P:{metrics['half_precision']*100:.0f}% R:{metrics['half_recall']*100:.0f}%) | "
              f"Dec-F1={metrics['dec_f1']*100:.1f}% (P:{metrics['dec_precision']*100:.0f}% R:{metrics['dec_recall']*100:.0f}%) | "
              f"Score={composite_score*100:.1f}% | LR={cur_lr:.1e}")

        if composite_score > best_score:
            best_score = composite_score
            torch.save(model.state_dict(), best_ckpt_path)
            # Cũng lưu đè best_digit_clf.pt để tương thích ngay với app.py
            torch.save(model.state_dict(), out_dir / "best_digit_clf.pt")
            print(f"   ★ Cải thiện mô hình tốt nhất! Đã lưu checkpoint ({composite_score*100:.2f}%)")

    print("\n" + "="*70)
    print(f"✓ HUẤN LUYỆN HOÀN TẤT! Score tốt nhất: {best_score*100:.2f}%")
    print(f"Checkpoint đã lưu tại: {best_ckpt_path}")
    print("="*70)


if __name__ == "__main__":
    main()

import sys
import shutil
import time
from pathlib import Path
import torch

BASE_DIR = Path("/home/deist/Downloads/OCR/anh")
sys.path.append(str(BASE_DIR))

def log_stage(title):
    print("\n" + "="*70)
    print(f"🚀 {title}")
    print("="*70, flush=True)

def train_obb(epochs=60):
    log_stage(f"[1/4] HUẤN LUYỆN MODEL OBB (YOLOv8-OBB: Phát hiện xoay nghiêng)")
    from ultralytics import YOLO

    # Ưu tiên fine-tune từ best.pt hiện có nếu tồn tại, ngược lại từ yolov8n-obb.pt
    ckpt = BASE_DIR / "runs" / "obb" / "train" / "weights" / "best.pt"
    init_model = str(ckpt) if ckpt.exists() else "yolov8n-obb.pt"
    print(f"Khởi tạo OBB từ: {init_model}")

    model = YOLO(init_model)
    results = model.train(
        data=str(BASE_DIR / "yolo_dataset" / "data.yaml"),
        epochs=epochs,
        imgsz=640,
        batch=16,
        device=0,
        patience=25,
        project=str(BASE_DIR / "runs" / "obb"),
        name="train",
        exist_ok=True,
        cache=True,
        workers=4,
        verbose=True
    )
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"✓ OBB Training hoàn thành. Weights: {best_weights}")
    return best_weights

def train_mechanical_bbox(epochs=60):
    log_stage(f"[2/4] HUẤN LUYỆN MODEL BBOX CÔNG TƠ CƠ (yolov8n-digits-bbox)")
    from ultralytics import YOLO

    ckpt = BASE_DIR / "yolov8n-digits-bbox.pt"
    init_model = str(ckpt) if ckpt.exists() else "yolov8n.pt"
    print(f"Khởi tạo Mechanical Bbox từ: {init_model}")

    model = YOLO(init_model)
    results = model.train(
        data=str(BASE_DIR / "yolo_digits_dataset" / "dataset.yaml"),
        epochs=epochs,
        imgsz=640,
        batch=16,
        device=0,
        patience=25,
        project=str(BASE_DIR / "runs" / "detect"),
        name="yolov8n_digits_bbox",
        exist_ok=True,
        cache=True,
        workers=4,
        verbose=True
    )
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    target = BASE_DIR / "yolov8n-digits-bbox.pt"
    if best_weights.exists():
        shutil.copy(best_weights, target)
        print(f"✓ Mechanical Bbox hoàn thành. Đã lưu đè: {target}")
    return target

def train_electronic_bbox(epochs=60):
    log_stage(f"[3/4] HUẤN LUYỆN MODEL BBOX CÔNG TƠ ĐIỆN TỬ (yolov8n-electronic-bbox)")
    from ultralytics import YOLO

    ckpt = BASE_DIR / "yolov8n-electronic-bbox.pt"
    init_model = str(ckpt) if ckpt.exists() else "yolov8n.pt"
    print(f"Khởi tạo Electronic Bbox từ: {init_model}")

    model = YOLO(init_model)
    results = model.train(
        data=str(BASE_DIR / "yolo_electronic_dataset" / "dataset.yaml"),
        epochs=epochs,
        imgsz=640,
        batch=16,
        device=0,
        patience=25,
        project=str(BASE_DIR / "runs" / "detect"),
        name="yolov8n_electronic_bbox",
        exist_ok=True,
        cache=True,
        workers=4,
        verbose=True
    )
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    target = BASE_DIR / "yolov8n-electronic-bbox.pt"
    if best_weights.exists():
        shutil.copy(best_weights, target)
        print(f"✓ Electronic Bbox hoàn thành. Đã lưu đè: {target}")
    return target

def train_electronic_ocr(epochs=60):
    log_stage(f"[4/4] HUẤN LUYỆN MODEL OCR CÔNG TƠ ĐIỆN TỬ (PARSeq: best_parseq_elec.pt)")
    import torch.nn as nn
    import torch.optim as optim
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
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        self.samples.append((parts[0], parts[1]))
            self.transform = transform

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            fname, label = self.samples[idx]
            img = Image.open(self.img_dir / fname).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Thiết bị train: {device}")

    model = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).to(device)
    ckpt = BASE_DIR / "best_parseq_elec.pt"
    if ckpt.exists():
        try:
            model.load_state_dict(torch.load(ckpt, map_location=device))
            print(f"Đã nạp weights hiện có từ {ckpt} để fine-tune tiếp.")
        except Exception as e:
            print(f"Không thể load state_dict ({e}), train từ baseline PARSeq.")

    train_transform = transforms.Compose([
        transforms.Resize((32, 128), transforms.InterpolationMode.BICUBIC),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    val_transform = transforms.Compose([
        transforms.Resize((32, 128), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    train_ds = PARSeqDataset(BASE_DIR / 'parseq_data/train/gt.txt', BASE_DIR / 'parseq_data/train/images', train_transform)
    val_ds = PARSeqDataset(BASE_DIR / 'parseq_data/val/gt.txt', BASE_DIR / 'parseq_data/val/images', val_transform)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    print(f"PARSeq: {len(train_ds)} train samples, {len(val_ds)} val samples.")

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-4, epochs=epochs, steps_per_epoch=len(train_loader)
    )

    best_acc = 0.0
    target = BASE_DIR / "best_parseq_elec.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_correct = 0
        train_total = 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            loss = model.training_step((imgs, labels), 0)
            loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                logits = model(imgs)
                pred = logits.softmax(-1)
                label_preds, _ = model.tokenizer.decode(pred)
                for pred_str, true_str in zip(label_preds, labels):
                    if pred_str == true_str:
                        train_correct += 1
                    train_total += 1

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

        acc = correct / total if total > 0 else 0
        train_acc = train_correct / train_total if train_total > 0 else 0
        is_best = acc > best_acc
        if is_best:
            best_acc = acc
            torch.save(model.state_dict(), target)

        if epoch % 5 == 0 or is_best or epoch == epochs:
            print(f"Epoch {epoch:2d}/{epochs} | Train Acc: {train_acc*100:.2f}% | Val Acc: {acc*100:.2f}% {'★ BEST' if is_best else ''}")

    print(f"✓ PARSeq Electronic OCR hoàn thành. Best Val Acc: {best_acc*100:.2f}%. Đã lưu: {target}")
    return target

def main():
    start_time = time.time()
    print("BẮT ĐẦU HUẤN LUYỆN TOÀN BỘ 4 MÔ HÌNH...")

    train_obb(epochs=60)
    train_mechanical_bbox(epochs=60)
    train_electronic_bbox(epochs=60)
    train_electronic_ocr(epochs=60)

    elapsed = time.time() - start_time
    print("\n" + "="*70)
    print(f"🎉 TẤT CẢ 4 MÔ HÌNH ĐÃ HUẤN LUYỆN THÀNH CÔNG trong {elapsed/60:.1f} phút!")
    print("="*70)

if __name__ == "__main__":
    main()

import torch, cv2, os, sys
from PIL import Image
from pathlib import Path
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).to(device).eval()
img_transform = transforms.Compose([
    transforms.Resize((32, 128), transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

val_dir = Path("/home/deist/Downloads/OCR/anh/parseq_mech_data/val")
gt_file = val_dir / "gt.txt"
img_dir = val_dir / "images"

correct = 0
total = 0

with open(gt_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        fname, label = line.split("\t")
        img_path = img_dir / fname
        if not img_path.exists(): continue
        
        total += 1
        img = Image.open(img_path).convert("RGB")
        inp = img_transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = parseq(inp)
            pred = logits.softmax(-1)
            pred_label, _ = parseq.tokenizer.decode(pred)
            pred_label = pred_label[0]
            
        if pred_label == label:
            correct += 1

print(f"Pretrained PARSeq Baseline on Mechanical: {correct}/{total} ({correct/total*100:.2f}%)")

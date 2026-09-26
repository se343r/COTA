import torch
import sys
sys.path.append("/home/deist/Downloads/OCR/anh/parseq")
from pathlib import Path
from PIL import Image
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).to(device).eval()
model.load_state_dict(torch.load("/home/deist/Downloads/OCR/anh/best_parseq_mech.pt", map_location=device))

val_transform = transforms.Compose([
    transforms.Resize((32, 128), transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

val_dir = Path("/home/deist/Downloads/OCR/anh/parseq_mech_data/val")
gt_file = val_dir / "gt.txt"
img_dir = val_dir / "images"

c = 0
with open(gt_file, "r") as f:
    for line in f:
        fname, label = line.strip().split('\t')
        img_path = img_dir / fname
        img = Image.open(img_path).convert('RGB')
        inp = val_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(inp).softmax(-1)
            pred_str, _ = model.tokenizer.decode(pred)
            pred_str = pred_str[0]
            print(f"File: {fname} | Expected: {label} | Pred: {pred_str}")
            if c > 10: break
            c += 1

import torch
from PIL import Image
from pathlib import Path
from torchvision import transforms

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    model = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).to(device)
    model.load_state_dict(torch.load("best_parseq_elec.pt", map_location=device))
    model.eval()
    
    img_transform = transforms.Compose([
        transforms.Resize((32, 128), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    
    gt_file = Path("parseq_data/gt.txt")
    img_dir = Path("parseq_data/images")
    
    lines = gt_file.read_text(encoding="utf-8").strip().split('\n')
    correct = 0
    total = len(lines)
    
    with torch.no_grad():
        for line in lines:
            if not line: continue
            fname, label = line.split('\t')
            img_path = img_dir / fname
            img = Image.open(img_path).convert('RGB')
            x = img_transform(img).unsqueeze(0).to(device)
            
            logits = model(x)
            pred = logits.softmax(-1)
            label_pred, _ = model.tokenizer.decode(pred)
            label_pred = label_pred[0]
            
            if label_pred == label:
                correct += 1
                
    acc = correct / total if total > 0 else 0
    print(f"Accuracy of FINE-TUNED PARSeq on all 191 electronic meters: {correct}/{total} = {acc*100:.2f}%")

if __name__ == '__main__':
    main()

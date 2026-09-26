import torch, sys
sys.path.append("/home/deist/Downloads/OCR/anh/parseq")
model = torch.hub.load('baudm/parseq', 'parseq', pretrained=True)
for name, module in model.named_children():
    print(name)

import torch, sys
sys.path.append("/home/deist/Downloads/OCR/anh/parseq")
model = torch.hub.load('baudm/parseq', 'parseq', pretrained=True)
for name, mod in model.named_parameters():
    print(name)
    break

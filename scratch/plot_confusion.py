import json, glob, random
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

import sys
sys.path.append(".")
from train_digit_classifier import load_clean_samples, split_crop_ids, DigitDataset, TinyDigitCNN, LABELS, LABEL2IDX
from torch.utils.data import DataLoader

json_glob = "ocr_dataset/labels/*.json"
crop_images_dir = "ocr_dataset/crops"
model_path = "output_digit_clf/best_digit_clf.pt"

samples, crop_ids = load_clean_samples(json_glob)
_, val_ids = split_crop_ids(crop_ids)

print(f"Total samples: {len(samples)}")
print(f"Total val ids: {len(val_ids)}")

# Correct argument order: samples, crop_images_dir, allowed_crop_ids, train=False
val_dataset = DigitDataset(samples, crop_images_dir, val_ids, train=False, img_size=32)
print(f"Val dataset size: {len(val_dataset)}")

val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TinyDigitCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

cm = np.zeros((10, 10), dtype=int)
y_true = []
y_pred = []

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(dim=1)
        for p, l in zip(preds.tolist(), labels.tolist()):
            cm[l, p] += 1
            y_pred.append(p)
            y_true.append(l)

print(f"len(y_true): {len(y_true)}, len(y_pred): {len(y_pred)}")
print(f"y_true[:10]: {y_true[:10]}")
print(f"y_pred[:10]: {y_pred[:10]}")

fig, ax = plt.subplots(figsize=(8, 8))
cax = ax.matshow(cm, cmap=plt.cm.Blues)
fig.colorbar(cax)

for i in range(10):
    for j in range(10):
        ax.text(j, i, str(cm[i, j]), va='center', ha='center',
                color='white' if cm[i, j] > cm.max()/2 else 'black')

ax.set_xticks(np.arange(10))
ax.set_yticks(np.arange(10))
ax.set_xticklabels(LABELS)
ax.set_yticklabels(LABELS)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Validation Confusion Matrix (Baseline v1)', pad=20)
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved to confusion_matrix.png")

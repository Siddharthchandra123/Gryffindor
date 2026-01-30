import albumentations as A
import matplotlib.pyplot as plt
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from torchmetrics import JaccardIndex
import albumentations as A


# print(img.shape, mask.shape)
# plt.figure(figsize=(8,4))
# plt.subplot(1,2,1); plt.title("Image"); plt.imshow(img.permute(1,2,0))
# plt.subplot(1,2,2); plt.title("Mask"); plt.imshow(mask)
# plt.show()



# src/train.py
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

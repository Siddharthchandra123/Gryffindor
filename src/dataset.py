# src/dataset.py
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

# Compress real labels -> contiguous class IDs
# Real dataset labels discovered:
# [0, 1, 2, 3, 27, 39]
LABEL_MAP = {
    0: 0,    # background
    1: 1,
    2: 2,
    3: 3,
    27: 4,
    39: 5
}

NUM_CLASSES = 6


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        # load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # load mask (grayscale)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # remap labels -> contiguous IDs
        final_mask = np.zeros_like(mask, dtype=np.uint8)
        for raw_label, new_label in LABEL_MAP.items():
            final_mask[mask == raw_label] = new_label

        mask = final_mask

        # apply transforms (NumPy only)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # convert to torch tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).long()

        return image, mask

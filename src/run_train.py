# src/run_train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from dataset import SegmentationDataset
from transform import get_train_transforms, get_val_transforms
from model import get_model
from training import train_one_epoch
from eval import evaluate
from utils import save_model

NUM_CLASSES = 6
EPOCHS = 10
BATCH_SIZE = 4
LR = 1e-3

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# datasets
train_ds = SegmentationDataset(
    r"data\Offroad_Segmentation_Training_Dataset\train\Color_Images",
    r"data\Offroad_Segmentation_Training_Dataset\train\Segmentation",
    transform=get_train_transforms()
)

val_ds = SegmentationDataset(
    r"data\Offroad_Segmentation_Training_Dataset\val\Color_Images",
    r"data\Offroad_Segmentation_Training_Dataset\val\Segmentation",
    transform=get_val_transforms()
)

# loaders
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# model
model = get_model(num_classes=NUM_CLASSES).to(device)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# combined loss
dice_loss = smp.losses.DiceLoss(mode="multiclass", ignore_index=0)
ce_loss = nn.CrossEntropyLoss(ignore_index=0)

def loss_fn(outputs, targets):
    return dice_loss(outputs, targets) + ce_loss(outputs, targets)

# training loop
best_iou = 0.0

for epoch in range(EPOCHS):
    loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
    iou = evaluate(model, val_loader, device, num_classes=NUM_CLASSES)

    print(f"Epoch {epoch+1} | Loss: {loss:.4f} | IoU: {iou:.4f}")

    if iou > best_iou:
        best_iou = iou
        save_model(model, "outputs/models/best_model.pth")

print("Training complete. Best IoU:", best_iou)

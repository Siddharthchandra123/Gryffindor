# src/evaluate.py
from torchmetrics import JaccardIndex
import torch

def evaluate(model, loader, device, num_classes):
    model.eval()
    iou = JaccardIndex(
        task="multiclass",
        num_classes=num_classes,
        ignore_index=0  # ignore background
    ).to(device)

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images).argmax(1)
            iou.update(preds, masks)

    iou_value = iou.compute()

    # NaN-safe return
    if torch.isnan(iou_value):
        return 0.0

    return iou_value.item()

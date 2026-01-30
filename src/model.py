# src/model.py
import segmentation_models_pytorch as smp

def get_model(num_classes):
    return smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes
    )

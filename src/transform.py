import albumentations as A

def get_train_transforms():
    return A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5)
    ])


def get_val_transforms():
    return A.Compose([
        A.Resize(256, 256),
    ])
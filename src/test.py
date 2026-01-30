import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset import SegmentationDataset
from model import get_model
from transform import get_val_transforms

# ---------------- CONFIG ----------------
NUM_CLASSES = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_IMG_DIR = r"data/Offroad_Segmentation_testImages/Color_Images"
OUT_DIR = r"outputs/test_predictions"
MODEL_PATH = r"outputs/models/best_model.pth"

os.makedirs(OUT_DIR, exist_ok=True)
# ----------------------------------------


# Dummy dataset (masks not used in testing)
test_ds = SegmentationDataset(
    image_dir=TEST_IMG_DIR,
    mask_dir=TEST_IMG_DIR,   # dummy, not used
    transform=get_val_transforms()
)

test_loader = DataLoader(
    test_ds,
    batch_size=1,
    shuffle=False
)

# Load model
model = get_model(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# print("Running inference on test images...")

# # Inference loop
# with torch.no_grad():
#     for idx, (images, _) in enumerate(test_loader):
#         images = images.to(DEVICE)

#         outputs = model(images)
#         preds = outputs.argmax(dim=1).cpu().numpy()[0]

#         save_path = os.path.join(OUT_DIR, f"pred_{idx:04d}.png")
#         cv2.imwrite(save_path, preds.astype(np.uint8))

# print("✅ Testing complete")
# print("Predictions saved to:", OUT_DIR)

import os
import cv2
import matplotlib.pyplot as plt
from glob import glob

# get first image that actually exists
image_files = glob(TEST_IMG_DIR + "/*")
assert len(image_files) > 0, "No images found in test directory"

img_path = image_files[0]
pred_path = "outputs/test_predictions/pred_0000.png"

# load image
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Failed to load image: {img_path}")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# load prediction
pred = cv2.imread(pred_path, 0)
if pred is None:
    raise FileNotFoundError(f"Failed to load prediction: {pred_path}")

# show
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title("Input Image")
plt.imshow(img)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Prediction")
plt.imshow(pred, cmap="tab10")
plt.axis("off")

plt.show()

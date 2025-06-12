import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.advanced_unet import UNET
from unet_dataset import PascalVOCDataset
from combined_loss import BCEDiceLoss
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# 🔧 Hyperparameters
ROOT_DIR = "VOC2012_train_val"
EPOCHS = 30
START_EPOCH = 1
BATCH_SIZE = 8
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_INTERVAL = 5
SAVE_DIR = "models/advanced_checkpoints"

# 🧠 Model
model = UNET(in_channels=3, out_channels=1).to(DEVICE)
if START_EPOCH > 1:
    ckpt_path = os.path.join(SAVE_DIR, f"unet_epoch_{START_EPOCH - 1}.pth")
    print(f"🔁 Loading checkpoint from: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

# 📉 Loss & Optimizer
criterion = BCEDiceLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# 🔁 Transforms
transform = A.Compose([
    A.Resize(height=256, width=256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
    ToTensorV2(),
])

# 📦 Dataset & Loader
train_dataset = PascalVOCDataset(
    root_dir=ROOT_DIR,
    split="train",
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Training Loop
os.makedirs(SAVE_DIR, exist_ok=True)
for epoch in range(START_EPOCH, EPOCHS + 1):
    model.train()
    epoch_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch [{epoch}/{EPOCHS}]")

    for images, masks in loop:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / len(train_loader)
    print(f"\n Epoch {epoch} complete - Avg Loss: {avg_loss:.4f}")

    if epoch % SAVE_INTERVAL == 0 or epoch == EPOCHS:
        save_path = os.path.join(SAVE_DIR, f"unet_epoch_{epoch}.pth")
        torch.save(model.state_dict(), save_path)
        print(f" Checkpoint saved at {save_path}")

# Final save
final_path = os.path.join(SAVE_DIR, "unet_final.pth")
torch.save(model.state_dict(), final_path)
print(f"\n🎉 Training completed. Final model saved at: {final_path}")

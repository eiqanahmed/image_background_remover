import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class PascalVOCDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Image and mask directories
        self.image_dir = os.path.join(root_dir, "JPEGImages")
        self.mask_dir = os.path.join(root_dir, "SegmentationClass")
        split_file = os.path.join(root_dir, "ImageSets", "Segmentation", f"{split}.txt")

        # Load image IDs from split
        with open(split_file, "r") as f:
            self.image_ids = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        mask_path = os.path.join(self.mask_dir, f"{image_id}.png")

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        # Convert to binary mask (any non-zero value is foreground)
        mask = (mask > 0).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].unsqueeze(0)  # Add channel dim for binary mask

        return image, mask

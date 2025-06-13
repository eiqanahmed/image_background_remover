import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from models.advanced_unet import UNET
import os

MODEL_PATH = "models/advanced_checkpoints/unet_epoch_5.pth"
IMG_PATH = "custom_input/white_bird.jpg"
OUTPUT_PATH = "output/white_cleaned_bird.png"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = (256, 256)

# Loading the model
model = UNET(in_channels=3, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Image transform
transform = T.Compose([
    T.Resize(IMAGE_SIZE),
    T.ToTensor(),
])

# Loading and preprocessing the input image
original = Image.open(IMG_PATH).convert("RGB")
original_size = original.size
input_tensor = transform(original).unsqueeze(0).to(DEVICE)

# Predicting the mask
with torch.no_grad():
    output = model(input_tensor)
    prob_mask = torch.sigmoid(output).squeeze().cpu().numpy()
    binary_mask = (prob_mask > 0.5).astype(np.uint8)

# Resizing the predicted mask back to original
resized_mask = Image.fromarray(binary_mask * 255).resize(original_size, Image.NEAREST)

# Apply alpha mask for background transparency
original_rgba = original.convert("RGBA")
mask_array = np.array(resized_mask)
alpha = (mask_array > 0).astype(np.uint8) * 255
rgba_array = np.array(original_rgba)
rgba_array[:, :, 3] = alpha

# Saving the result
cleaned = Image.fromarray(rgba_array)
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
cleaned.save(OUTPUT_PATH)
print(f" Processed image saved at {OUTPUT_PATH}")



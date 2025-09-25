import os
import torch
from torchvision.utils import save_image
from datetime import datetime

def unnormalize(img, mean, std):
    if img.dim() == 3:
        mean_t = img.new_tensor(mean)[:, None, None]
        std_t  = img.new_tensor(std)[:, None, None]
    else:
        mean_t = img.new_tensor(mean)[None, :, None, None]
        std_t  = img.new_tensor(std)[None, :, None, None]
    return img * std_t + mean_t

def save_generated_images(x_gen, out_dir="aenf_output", num_images=5,
                          mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
    """
    Save a batch of generated images to disk.
    x_gen: Tensor [B,C,H,W] in normalized space.
    """
    os.makedirs(out_dir, exist_ok=True)
    datetime_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Make sure we don't index out of bounds
    n = min(num_images, x_gen.size(0))

    for i in range(n):
        img = x_gen[i].cpu()
        img_vis = unnormalize(img, mean, std).clamp(0,1)
        filename = os.path.join(out_dir, f"gen_{datetime_now}_{i+1}.png")
        save_image(img_vis, filename)
        print(f"Saved {filename}")

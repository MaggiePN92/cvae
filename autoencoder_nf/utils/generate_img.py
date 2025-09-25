import os
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from datetime import datetime


def unnormalize(img, mean, std):
    # img: (B,C,H,W) or (C,H,W) in normalized space
    if img.dim() == 3:
        mean_t = img.new_tensor(mean)[:, None, None]
        std_t  = img.new_tensor(std)[:, None, None]
    else:
        mean_t = img.new_tensor(mean)[None, :, None, None]
        std_t  = img.new_tensor(std)[None, :, None, None]
    return img * std_t + mean_t


def generate_img(
    conditioned_model, z_dim, device, 
    out_dir="generated_imgs", tau=0.5, 
    seed=None):

    conditioned_model.eval()
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)

    os.makedirs(out_dir, exist_ok=True)

    #generate_img(trained_model, z_dim, device, seed=42)
    datetime_now = datetime.now().strftime("%m-%d_%H-%M-%S")
    filename = os.path.join(out_dir, f"img-{datetime_now}.png")
    with torch.no_grad():
        z = torch.randn(1, z_dim, generator=g, device=device)
        c_raw = torch.randint(0, 10, (1, 6), generator=g, device=device, dtype=torch.float32)
        c_raw[0, 1] = 9
        c = torch.softmax(c_raw / tau, dim=1)

        zc = torch.cat([z, c], dim=1)
        x_hat = conditioned_model.decoder(zc) # [1,3,H,W]
        x_hat_vis = unnormalize(x_hat.cpu(), mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        save_image(x_hat_vis.clamp(0,1), filename)

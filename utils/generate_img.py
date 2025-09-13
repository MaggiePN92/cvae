import os
import torch
import torch.nn.functional as F
from torchvision.utils import save_image


def generate_img(
        conditioned_model, z_dim, device, 
        out_dir="generated_imgs", tau=0.5, 
        seed=None
    ):
    conditioned_model.eval()
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)

    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        # 1) z ~ N(0, I)
        z = torch.randn(1, z_dim, generator=g, device=device)

        # 2) preference vector [x1, 9, x2, x3, x4, x5]
        c_raw = torch.randint(0, 10, (1, 6), generator=g, device=device, dtype=torch.float32)
        c_raw[0, 1] = 9
        c = torch.softmax(c_raw / tau, dim=1)

        # 3) concat and decode
        zc = torch.cat([z, c], dim=1)
        x_hat = conditioned_model.decoder(zc).squeeze(0).cpu()

    # 4) save image
    # save_image expects (C,H,W) or (B,C,H,W); clamp for safety
    filename = os.path.join(
        out_dir, f"moira_{seed or torch.randint(0, 1_000_000, (1,)).item()}.png")
    save_image(x_hat.clamp(0, 1), filename)
    print(f"Saved {filename}")

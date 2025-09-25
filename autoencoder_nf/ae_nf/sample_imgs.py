import torch


@torch.no_grad()
def sample_images(ae, flow, c_raw, num=1, device="cuda"):
    """
    Sample images from trained AE + NF model.

    ae: trained autoencoder (CondAE)
    flow: trained flow (LatentFlow)
    c_raw: raw class/preferences tensor [B, n_classes] or [1, n_classes]
    num: how many images to generate
    device: device for tensors
    """
    ae.eval()
    flow.eval()

    # broadcast c_raw if only one condition is given
    if c_raw.size(0) == 1 and num > 1:
        c_raw = c_raw.expand(num, -1)

    c_vec = ae.scale_class_vec(c_raw.to(device)) # [num, n_classes]
    z = flow.sample(num_samples=num, cond=c_vec) # [num, z_dim]
    x_hat = ae.decode(z, c_vec) # [num, C, H, W]
    return x_hat

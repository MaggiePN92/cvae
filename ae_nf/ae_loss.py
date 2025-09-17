import torch.nn.functional as F

def ae_loss(x, x_hat, z, lam=1e-3):
    recon = F.l1_loss(x_hat, x, reduction="mean")
    reg = lam * z.pow(2).mean()
    return recon + reg, recon, reg

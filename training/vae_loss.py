import torch
import torch.nn.functional as F


def vae_loss(x, x_hat, mu, logvar, beta, logdet=None):
    recon = F.l1_loss(x_hat, x, reduction="mean")
    base_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    if logdet is not None:
        kl = base_kl - logdet.mean()
    else:
        kl = base_kl
    total = recon + beta * kl
    return total, recon, kl

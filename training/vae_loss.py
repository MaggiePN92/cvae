import torch
import torch.nn.functional as F


def vae_loss(x, x_hat, mean, logvar, beta):
    """
    Inputs:
        x       : [torch.tensor] Original sample
        x_hat   : [torch.tensor] Reproduced sample
        mean    : [torch.tensor] Mean mu of the variational posterior given sample x
        logvar  : [torch.tensor] log of the variance sigma^2 of the variational posterior given sample x
    """

    # Recontruction loss
    recon = F.mse_loss(x_hat, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    loss = recon + beta * kl
    return loss, recon, kl 

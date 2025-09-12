import torch
import torch.nn.functional as F


def vae_loss(x, x_hat, mean, logvar):
    """
    Inputs:
        x       : [torch.tensor] Original sample
        x_hat   : [torch.tensor] Reproduced sample
        mean    : [torch.tensor] Mean mu of the variational posterior given sample x
        logvar  : [torch.tensor] log of the variance sigma^2 of the variational posterior given sample x
    """

    # Recontruction loss
    reproduction_loss = F.mse_loss(x_hat, x, reduction="sum")

    # KL divergence
    KL_divergence = (-1/2) * (torch.sum((1 + logvar - torch.pow(mean, 2) - torch.exp(logvar))))

    # Get the total loss
    loss = reproduction_loss + KL_divergence
    return loss

import torch

def reparameterize_gaussian(mean, logvar):
    # reparameterization trick
    eps = torch.randn_like(mean)
    std = torch.exp(0.5 * logvar)  # convert logvar to std
    z = mean + std * eps
    return z
import torch
import torch.nn.functional as F

def vae_loss(x, x_hat, mean, logvar, beta):
    # reconstruction loss
    recon = F.mse_loss(x_hat, x, reduction="mean")
    
    # kl divergence
    kl = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    
    # total loss
    loss = recon + beta * kl
    
    return loss, recon, kl
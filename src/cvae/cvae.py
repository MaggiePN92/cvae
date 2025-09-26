import torch
import torch.nn as nn
from .cdecoder import CDecoder
from .cencoder import CEncoder
from .reparameterize_gaussian import reparameterize_gaussian

class CVAE(nn.Module):
    # conditional vae for car images
    def __init__(self, z_dim, condition_dim=2, n_channels=3, img_size=[96,128]):
        super().__init__()
        self.z_dim = z_dim
        self.condition_dim = condition_dim  # moira and ferdinando scores
        self.n_channels = n_channels
        self.img_size = img_size

        # encoder
        self.encoder = CEncoder(z_dim, n_channels, img_size, condition_dim)
        latent_hw = self.encoder.latent_hw
        
        # decoder
        self.decoder = CDecoder(z_dim, condition_dim, n_channels, img_size, latent_hw, start_ch=128)

    def forward(self, x, c):
        # encode
        mean, logvar = self.encoder(x, c)
        
        # sample z
        z = reparameterize_gaussian(mean, logvar)
        
        # decode
        z_cat = torch.cat([z, c], dim=1)
        x_hat = self.decoder(z_cat)
        
        return x_hat, mean, logvar, c
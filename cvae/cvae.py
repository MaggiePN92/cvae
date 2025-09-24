import torch
import torch.nn as nn
from .cdecoder import CDecoder
from .cencoder import CEncoder
from .reparameterize_gaussian import reparameterize_gaussian
from .latent_flow import LatentFlow


class CVAE(nn.Module):
    """ Conditional Variational Auto-Encoder class. """

    def __init__(self, z_dim, n_classes, n_channels=1, img_size=[96,128],
                 return_feats = True):
        super().__init__()
        self.z_dim = z_dim
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.img_size = img_size
        self.return_feats = return_feats

        self.flow = LatentFlow(
            dim=z_dim, cond_dim=n_classes, 
            hidden=64, K=2)
        
        self.encoder = CEncoder(
            z_dim=z_dim, n_channels=3, img_size=(96,128), ch=(64,128))
        
        self.decoder = CDecoder(
            z_dim=z_dim, n_classes=6, n_channels=3,
            img_size=(96,128),
            latent_hw=self.encoder.latent_hw,
            start_ch=192,
            skip_channels=[
                self.encoder.c1, self.encoder.c2],  # [64, 128]
        )

        # Add learnable class token
        self.cls_param = nn.Parameter(torch.zeros(n_classes, *img_size))

    def scale_class_vec(self, c, sharpness=0.5):
        c = torch.softmax(c / sharpness, dim=1) 
        return c

    def get_cls_emb(self, c):  # c: [B,6] in [0,1]
        # (B n) x (n H W) -> (B H W)
        emb = torch.einsum('bn,nhw->bhw', c, self.cls_param) # [B,H,W]
        return emb.unsqueeze(1)
     
    def forward(self, x, c):
        c_vec = self.scale_class_vec(c) # [B, n_classes] in [0,1]
        emb = self.get_cls_emb(c_vec) # [B,1,H,W]
        x4 = torch.cat([x, emb], dim=1) # [B, n_channels+1, H, W]

        mean, logvar, enc_feats = self.encoder(
            x4, return_feats=self.return_feats)
        z0 = reparameterize_gaussian(mean, logvar)

        z_cat = torch.cat([z0, c_vec], dim=1)
        x_hat = self.decoder(z_cat, enc_feats=enc_feats)
        return x_hat, mean, logvar

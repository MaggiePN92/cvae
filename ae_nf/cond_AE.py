import torch
import torch.nn as nn
from .cencoder import CEncoder
from .cdecoder import CDecoder


class CondAE(nn.Module):
    def __init__(self, z_dim, n_classes, n_channels=3, img_size=(96,128)):
        super().__init__()
        self.z_dim = z_dim
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.img_size = img_size

        self.encoder = CEncoder(z_dim=z_dim, n_channels=n_channels, img_size=img_size, ch=(64,128))
        self.decoder = CDecoder(z_dim=z_dim, n_classes=n_classes, n_channels=n_channels,
                                img_size=img_size, latent_hw=self.encoder.latent_hw, start_ch=192)

        # learnable spatial class embedding (keep if you like it)
        self.cls_param = nn.Parameter(torch.zeros(n_classes, *img_size))

    def scale_class_vec(self, c, sharpness=0.5):
        return torch.softmax(c / sharpness, dim=1)  # prefs -> soft probs

    def get_cls_emb(self, c):  # c: [B,n_classes]
        emb = torch.einsum('bn,nhw->bhw', c, self.cls_param)  # [B,H,W]
        return emb.unsqueeze(1)  # [B,1,H,W]

    def encode(self, x, c_vec):
        emb = self.get_cls_emb(c_vec)
        x_in = torch.cat([x, emb], dim=1)
        z = self.encoder(x_in)
        return z

    def decode(self, z, c_vec):
        zc = torch.cat([z, c_vec], dim=1)
        x_hat = self.decoder(zc)
        return x_hat

    def forward(self, x, c):
        c_vec = self.scale_class_vec(c)
        z = self.encode(x, c_vec)
        x_hat = self.decode(z, c_vec)
        return x_hat, z

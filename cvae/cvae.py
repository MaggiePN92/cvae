import torch
import torch.nn as nn
from .cdecoder import CDecoder
from .cencoder import CEncoder
from .reparameterize_gaussian import reparameterize_gaussian


class CVAE(nn.Module):
    """ Conditional Variational Auto-Encoder class. """

    def __init__(self, z_dim, n_classes, n_channels=1, img_size=[96,128]):
        super().__init__()
        self.z_dim = z_dim
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.img_size = img_size

        self.encoder = CEncoder(z_dim, n_channels, img_size)
        latent_hw = self.encoder.latent_hw
        self.decoder = CDecoder(
            z_dim, n_classes, n_channels, 
            img_size, latent_hw, start_ch=128
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
        """
        Args:
            x   : [torch.Tensor] Image input of shape [batch_size, n_channels, *img_size]
            c   : [torch.Tensor] Class labels for x of shape [batch_size], where the class in indicated by a
        """
        c_vec = self.scale_class_vec(c) # [B, n_classes] in [0,1]
        emb = self.get_cls_emb(c_vec) # [B,1,H,W]
        x4 = torch.cat([x, emb], dim=1) # [B, n_channels+1, H, W]
        mean, logvar = self.encoder(x4)
        z = reparameterize_gaussian(mean, logvar) # [B, z_dim]
        z_cat = torch.cat([z, c_vec], dim=1) # [B, z_dim + n_classes]
        x_hat = self.decoder(z_cat) # [B, n_channels, H, W]
        return x_hat, mean, logvar, c_vec



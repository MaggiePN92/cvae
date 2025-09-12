import torch
import torch.nn as nn
import torch.nn.functional as F
from .cdecoder import CDecoder
from .cencoder import CEncoder
from .reparameterize_gaussian import reparameterize_gaussian


class CVAE(nn.Module):
    """ Conditional Variational Auto-Encoder class. """

    def __init__(self, z_dim, n_classes, n_channels=1, img_size=[28,28]):
        super().__init__()
        self.z_dim = z_dim
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.img_size = img_size

        self.encoder = CEncoder(z_dim, n_classes, n_channels)
        self.decoder = CDecoder(z_dim, n_classes, n_channels)

        # Add learnable class token
        self.cls_param = nn.Parameter(torch.zeros(n_classes, *img_size))

    def get_cls_emb(self, c):
        return self.cls_param[c].unsqueeze(1)

    def forward(self, x, c):
        """
        Args:
            x   : [torch.Tensor] Image input of shape [batch_size, n_channels, *img_size]
            c   : [torch.Tensor] Class labels for x of shape [batch_size], where the class in indicated by a
        """

        assert x.shape[1:] == (self.n_channels, *self.img_size), f'Expected input x of shape [batch_size, {[self.n_channels, *self.img_size]}], but got {x.shape}'
        assert c.shape[0] == x.shape[0], f'Inputs x and c must have same batch size, but got {x.shape[0]} and {c.shape[0]}'
        assert len(c.shape) == 1, f'Input c should have shape [batch_size], but got {c.shape}'

        # Get cls embedding
        cls_emb = self.get_cls_emb(c)

        # Concatenate cls embedding to the input
        x = torch.cat((x, cls_emb), dim=1)

        # Get the mean, logvar, and cls token from the encoder
        mean, logvar, cls_token = self.encoder(x)

        # Calculate the standard deviation. Note: in log-space, squareroot is divide by two
        std = torch.exp(logvar / 2)

        # Sample the latent using the reparameterization trick
        z = reparameterize_gaussian(mean, std)

        # Concatenate cls token to z
        z = torch.cat((z, F.softmax(cls_token, dim=1)), dim=1)

        # Get reconstructed x from the decoder
        x_hat = self.decoder(z)
        
        return x_hat, mean, logvar, cls_token

import torch
import torch.nn as nn
import torch.nn.functional as F


class CDecoder(nn.Module):
    """ Convolutional decoder for the CVAE. """

    def __init__(self, z_dim, n_classes, n_channels):
        super().__init__()
        feature_dim = 32*6*6
        self.linear = nn.Linear(z_dim + n_classes, feature_dim)
        self.deconv1 = nn.ConvTranspose2d(
            32, 16, kernel_size=3, stride=2)
        self.deconv2 = nn.ConvTranspose2d(
            16, n_channels, kernel_size=3, stride=2, output_padding=1)

    def forward(self, z : torch.Tensor) -> torch.Tensor:
        x_hat = F.relu(self.linear(z))
        # Resphaping the tensor for the cnn
        x_hat = x_hat.view(-1, 32, 6, 6)
        x_hat = F.relu(self.deconv1(x_hat))
        x_hat = torch.sigmoid(self.deconv2(x_hat))
        return x_hat

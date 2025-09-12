import torch
import torch.nn as nn
import torch.nn.functional as F


class CEncoder(nn.Module):
    """ Convolutional encoder for the CVAE. """

    def __init__(self, z_dim, n_classes, n_channels):
        super().__init__()
        feature_dim = 32 * 6 * 6
        self.conv1 = nn.Conv2d(n_channels + 1, 16, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.mean_fc = nn.Linear(feature_dim, z_dim)
        self.logvar_fc = nn.Linear(feature_dim, z_dim)
        self.cls_fc = nn.Linear(feature_dim, n_classes)

    def forward(self, x : torch.Tensor):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.flatten(1)
        mean = self.mean_fc(x)
        logvar = self.logvar_fc(x)
        cls_token = self.cls_fc(x)
        return mean, logvar, cls_token

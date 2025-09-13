import torch.nn as nn
import torch.nn.functional as F


class CEncoder(nn.Module):
    def __init__(self, z_dim, n_channels, img_size, ch=(16, 32)):
        super().__init__()
        H, W = img_size
        c1, c2 = ch

        # two downsampling convs (stride=2 halves H,W each time)
        self.conv1 = nn.Conv2d(n_channels + 1, c1, kernel_size=3, stride=2, padding=1) # H/2, W/2
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1) # H/4, W/4

        # feature dim after flattening
        h4, w4 = H // 4, W // 4
        feature_dim = c2 * h4 * w4

        self.mean_fc   = nn.Linear(feature_dim, z_dim)
        self.logvar_fc = nn.Linear(feature_dim, z_dim)

        self.latent_hw = (h4, w4)
        self.c2 = c2

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x_flat = x.flatten(1)
        mean   = self.mean_fc(x_flat)
        logvar = self.logvar_fc(x_flat)
        return mean, logvar
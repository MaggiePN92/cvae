import torch.nn as nn
import torch.nn.functional as F

class CEncoder(nn.Module):
    def __init__(self, z_dim, n_channels, img_size, ch=(16, 32)):
        super().__init__()
        H, W = img_size
        c1, c2 = ch

        self.conv1 = nn.Conv2d(n_channels + 1, c1, 3, stride=2, padding=1)  # H/2, W/2
        self.conv2 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)              # H/4, W/4

        h4, w4 = H // 4, W // 4
        # derive from actual modules to avoid mismatches
        self.c1 = self.conv1.out_channels
        self.c2 = self.conv2.out_channels
        feature_dim = self.c2 * h4 * w4

        self.mean_fc   = nn.Linear(feature_dim, z_dim)
        self.logvar_fc = nn.Linear(feature_dim, z_dim)

        self.latent_hw = (h4, w4)

    def forward(self, x, return_feats: bool = False):
        f1 = F.relu(self.conv1(x)) # [B, c1, H/2, W/2]
        f2 = F.relu(self.conv2(f1)) # [B, c2, H/4, W/4]
        x_flat = f2.flatten(1)
        mean   = self.mean_fc(x_flat)
        logvar = self.logvar_fc(x_flat)
        if return_feats:
            return mean, logvar, [f1, f2]
        return mean, logvar

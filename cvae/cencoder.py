import torch.nn as nn
import torch.nn.functional as F

class CEncoder(nn.Module):
    def __init__(self, z_dim, n_channels, img_size, condition_dim=2, ch=(16, 32)):
        super().__init__()
        H, W = img_size
        c1, c2 = ch
        self.condition_dim = condition_dim

        # conv layers
        self.conv1 = nn.Conv2d(n_channels, c1, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1)

        # calculate feature size
        h4, w4 = H // 4, W // 4
        feature_dim = c2 * h4 * w4

        # condition projection
        self.condition_proj = nn.Linear(condition_dim, feature_dim)
        
        # output layers
        self.mean_fc = nn.Linear(feature_dim, z_dim)
        self.logvar_fc = nn.Linear(feature_dim, z_dim)

        self.latent_hw = (h4, w4)
        self.c2 = c2

    def forward(self, x, c):
        # conv layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x_flat = x.flatten(1)
        
        # add condition
        c_proj = self.condition_proj(c)
        x_conditioned = x_flat + c_proj
        
        # get mean and logvar
        mean = self.mean_fc(x_conditioned)
        logvar = self.logvar_fc(x_conditioned)
        
        return mean, logvar
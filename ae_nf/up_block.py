import torch.nn as nn
import torch.nn.functional as F


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_ch, in_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(
            in_ch, out_ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(
            x, scale_factor=2, 
            mode='bilinear', 
            align_corners=False
        )
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

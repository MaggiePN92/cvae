import torch
import torch.nn as nn
import torch.nn.functional as F


class CDecoder(nn.Module):
    def __init__(self, z_dim, n_classes, n_channels, img_size, base_ch=32):
        super().__init__()
        H, W = img_size
        h4, w4 = H // 4, W // 4

        # project z and c
        self.proj = nn.Linear(z_dim + n_classes, base_ch * h4 * w4)

        # upsample with ConvTranspose2d (stride=2 doubles spatial dims)
        self.deconv1 = nn.ConvTranspose2d(base_ch, base_ch // 2, kernel_size=4, stride=2, padding=1) # H/2, W/2
        self.deconv2 = nn.ConvTranspose2d(base_ch // 2, base_ch // 4, kernel_size=4, stride=2, padding=1) # H,W

        self.to_img = nn.Conv2d(base_ch // 4, n_channels, kernel_size=3, padding=1)

        self.h4, self.w4 = h4, w4
        self.base_ch = base_ch

    def forward(self, z_cat):
        B = z_cat.size(0)

        # seed feature map
        x = self.proj(z_cat) # [B, base_ch*h4*w4]
        x = x.view(B, self.base_ch, self.h4, self.w4) # [B, base_ch, H/4, W/4]

        # two upsampling deconvs
        x = F.relu(self.deconv1(x)) # [B, base_ch//2, H/2, W/2]
        x = F.relu(self.deconv2(x)) # [B, base_ch//4, H, W]

        # final image projection
        x = torch.sigmoid(self.to_img(x)) # [B, n_channels, H, W]
        return x
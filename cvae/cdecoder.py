import torch.nn as nn
from .up_block import UpBlock
import torch.nn.functional as F


class CDecoder(nn.Module):
    def __init__(
            self, z_dim, n_classes, n_channels, 
            img_size, latent_hw, start_ch=128
        ):
        super().__init__()
        self.H, self.W = img_size
        self.h0, self.w0 = latent_hw  # e.g. (24,32) after two downs
        self.start_ch = start_ch
        self.proj = nn.Linear(
            z_dim + n_classes, start_ch * self.h0 * self.w0)
        
        # build enough up blocks to reach or exceed target
        ups = []
        ch = start_ch
        curH, curW = self.h0, self.w0
        while curH < self.H or curW < self.W:
            nxt = max(32, ch // 2)
            ups.append(UpBlock(ch, nxt))
            ch = nxt
            curH, curW = curH * 2, curW * 2
        self.up_blocks = nn.ModuleList(ups)

        self.to_img = nn.Conv2d(ch, n_channels, 3, padding=1)

    def forward(self, z_cat):
        b = z_cat.size(0)
        x = self.proj(z_cat).view(b, self.start_ch, self.h0, self.w0)
        for up in self.up_blocks:
            x = up(x)
        if x.shape[-2:] != (self.H, self.W):
            x = F.interpolate(
                x, size=(self.H, self.W), 
                mode='bilinear', align_corners=False
            )
        x = self.to_img(x)  # no sigmoid if training on normalized pixels
        return x

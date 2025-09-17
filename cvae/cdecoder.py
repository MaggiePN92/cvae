import torch.nn as nn
from .up_block import UpBlock
import torch.nn.functional as F
import torch 


class CDecoder(nn.Module):
    def __init__(self, z_dim, n_classes, n_channels, 
                 img_size, latent_hw, start_ch=128, skip_channels=None):
        super().__init__()
        self.H, self.W = img_size
        self.h0, self.w0 = latent_hw
        self.start_ch = start_ch

        self.proj = nn.Linear(z_dim + n_classes, start_ch * self.h0 * self.w0)

        # Build up path and RECORD stage out channels
        ups, out_chs = [], []
        ch = start_ch
        curH, curW = self.h0, self.w0
        while curH < self.H or curW < self.W:
            nxt = max(32, ch // 2)
            ups.append(UpBlock(ch, nxt))
            ch = nxt
            out_chs.append(ch)          # <<< IMPORTANT: record decoder stage output ch
            curH, curW = curH * 2, curW * 2
        self.up_blocks = nn.ModuleList(ups)

        # Skips
        self.skip_channels = list(skip_channels or [])
        sc = self.skip_channels[::-1]   # lowest→highest to match up path

        # Optional sanity check to catch mismatches early
        if len(sc) < len(self.up_blocks):
            # We'll only fuse as many stages as we have skips for.
            pass  # or raise if you require 1:1
        # Build 1x1 fusers for the fused stages
        self.fuse = nn.ModuleList([
            nn.Conv2d(out_chs[i] + sc[i], out_chs[i], kernel_size=1)
            for i in range(min(len(self.up_blocks), len(sc)))
        ])

        self.to_img = nn.Conv2d(ch, n_channels, 3, padding=1)

    def forward(self, z_cat, enc_feats=None):
        b = z_cat.size(0)
        x = self.proj(z_cat).view(b, self.start_ch, self.h0, self.w0)

        enc_rev = list(enc_feats)[::-1] if enc_feats else []

        for i, up in enumerate(self.up_blocks):
            x = up(x)
            if i < len(self.fuse) and i < len(enc_rev):
                skip = enc_rev[i]
                if skip.shape[-2:] != x.shape[-2:]:
                    skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, skip], dim=1)
                x = self.fuse[i](x)

        if x.shape[-2:] != (self.H, self.W):
            x = F.interpolate(x, size=(self.H, self.W), mode='bilinear', align_corners=False)
        return self.to_img(x)

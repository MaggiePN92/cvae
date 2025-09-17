import torch.nn as nn
from .up_block import UpBlock
import torch.nn.functional as F
import torch 


class CDecoder(nn.Module):
    def __init__(
            self, z_dim, n_classes, n_channels, 
            img_size, latent_hw, start_ch=128,
            skip_channels=None
        ):
        super().__init__()
        self.H, self.W = img_size
        self.h0, self.w0 = latent_hw  # e.g. (24,32) after two downs
        self.start_ch = start_ch
        self.proj = nn.Linear(
            z_dim + n_classes, start_ch * self.h0 * self.w0)
        
        # build enough up blocks to reach or exceed target
        ups, out_chs = [], []
        ch = start_ch
        curH, curW = self.h0, self.w0
        while curH < self.H or curW < self.W:
            nxt = max(32, ch // 2)
            ups.append(UpBlock(ch, nxt))
            ch = nxt
            curH, curW = curH * 2, curW * 2
        self.up_blocks = nn.ModuleList(ups)

        # Prepare 1x1 fusers for skips (match decoder stage i with reversed encoder feats)
        self.skip_channels = list(skip_channels or [])
        sc = self.skip_channels[::-1]  # lowest→highest to match up path
        self.fuse = nn.ModuleList()
        for i in range(min(len(self.up_blocks), len(sc))):
            self.fuse.append(
                nn.Conv2d(
                    out_chs[i] + sc[i], out_chs[i], 
                    kernel_size=1
                )
            )

        self.to_img = nn.Conv2d(ch, n_channels, 3, padding=1)

    def forward(self, z_cat : torch.Tensor, enc_feats=None):
        b = z_cat.size(0)
        x = self.proj(z_cat).view(b, self.start_ch, self.h0, self.w0)

        enc_rev = []
        if enc_feats:
            enc_rev = list(enc_feats)[::-1]  # [f2, f1]

        for i, up in enumerate(self.up_blocks):
            x = up(x)
            if i < len(self.fuse) and i < len(enc_rev):
                skip = enc_rev[i]
                if skip.shape[-2:] != x.shape[-2:]:
                    skip = F.interpolate(
                        skip, 
                        size=x.shape[-2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                x = torch.cat([x, skip], dim=1)
                x = self.fuse[i](x)

        if x.shape[-2:] != (self.H, self.W):
            x = F.interpolate(
                x, 
                size=(self.H, self.W), 
                mode='bilinear', 
                align_corners=False
            )
        return self.to_img(x)

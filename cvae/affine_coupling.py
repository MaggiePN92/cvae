import torch 
import torch.nn as nn


class AffineCoupling(nn.Module):
    def __init__(self, dim, hidden=64, mask_even=True, cond_dim=0):
        super().__init__()
        self.mask_even = mask_even
        in_dim = dim//2 + cond_dim
        self.s = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, dim//2))
        self.t = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, dim//2))

    def forward(self, z, cond=None):
        # z: [B, dim], cond: [B, cond_dim] or None
        B, D = z.shape
        # even/odd indicies 
        idx = torch.arange(D, device=z.device) % 2
        if self.mask_even:
            a, b = z[:, idx==0], z[:, idx==1]  # keep a, transform b
        else:
            a, b = z[:, idx==1], z[:, idx==0]
        inp = a if cond is None else torch.cat([a, cond], dim=1)
        s, t = self.s(inp), self.t(inp)
        b_out = b * torch.exp(s) + t
        logdet = s.sum(dim=1)  # per-sample
        if self.mask_even:
            z_out = torch.zeros_like(z); z_out[:, idx==0] = a; z_out[:, idx==1] = b_out
        else:
            z_out = torch.zeros_like(z); z_out[:, idx==1] = a; z_out[:, idx==0] = b_out
        return z_out, logdet

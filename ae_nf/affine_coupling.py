import torch
import torch.nn as nn

class AffineCoupling(nn.Module):
    def __init__(self, dim, hidden=64, mask_even=True, cond_dim=0):
        super().__init__()
        self.mask_even = mask_even
        in_dim = dim//2 + cond_dim
        self.s = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, dim//2))
        self.t = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, dim//2))

    def _split(self, z):
        B, D = z.shape
        idx = torch.arange(D, device=z.device) % 2
        if self.mask_even:
            a, b = z[:, idx==0], z[:, idx==1]  # keep a, transform b
            even_mask = idx==0
        else:
            a, b = z[:, idx==1], z[:, idx==0]
            even_mask = idx==1
        return a, b, even_mask

    def forward(self, z, cond=None):
        a, b, even_mask = self._split(z)
        inp = a if cond is None else torch.cat([a, cond], dim=1)
        s, t = self.s(inp), self.t(inp)
        b_out = b*torch.exp(s) + t
        logdet = s.sum(dim=1)
        z_out = torch.zeros_like(z)
        if self.mask_even:
            z_out[:, even_mask] = a
            z_out[:, ~even_mask] = b_out
        else:
            z_out[:, ~even_mask] = a
            z_out[:, even_mask] = b_out
        return z_out, logdet

    def inverse(self, z_out, cond=None):
        a, b_out, even_mask = self._split(z_out)
        inp = a if cond is None else torch.cat([a, cond], dim=1)
        s, t = self.s(inp), self.t(inp)
        b = (b_out - t)*torch.exp(-s)
        z = torch.zeros_like(z_out)
        if self.mask_even:
            z[:, even_mask] = a
            z[:, ~even_mask] = b
        else:
            z[:, ~even_mask] = a
            z[:, even_mask] = b
        return z

from .affine_coupling import AffineCoupling
import torch.nn as nn


class LatentFlow(nn.Module):
    def __init__(self, dim, cond_dim=0, hidden=64, K=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            AffineCoupling(
                dim, hidden, 
                mask_even=(k%2==0), 
                cond_dim=cond_dim) for k in range(K)
        ])
        
    def forward(self, z, cond=None):
        logdet = z.new_zeros(z.size(0))
        for blk in self.blocks:
            z, ld = blk(z, cond)
            logdet = logdet + ld
        return z, logdet

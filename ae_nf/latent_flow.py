from .affine_coupling import AffineCoupling
import torch.nn as nn
import math
import torch 


class LatentFlow(nn.Module):
    def __init__(self, dim, cond_dim=0, hidden=64, K=2):
        super().__init__()
        self.dim = dim 
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

    def log_prob(self, z, cond=None):
        u, logdet = self.forward(z, cond) # z -> base u
        # log N(u;0,I) = -0.5*(||u||^2 + D log(2pi))
        log_pu = -0.5*(u.pow(2).sum(dim=1) + self.dim*math.log(2*math.pi))
        return log_pu + logdet  
    
    @torch.no_grad()
    def inverse(self, u, cond=None):
        for blk in reversed(self.blocks):
            u = blk.inverse(u, cond)
        return u

    @torch.no_grad()
    def sample(self, num_samples, cond, device=None):
        if device is None: device = cond.device
        u = torch.randn(num_samples, self.dim, device=device)
        # if cond is [1,C] and num_samples>1, expand it
        if cond.size(0)==1 and num_samples>1:
            cond = cond.expand(num_samples, -1)
        return self.inverse(u, cond)

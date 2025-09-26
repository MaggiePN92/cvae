import torch
import torch.nn as nn
import torch.nn.functional as F

class VQEmbedding(nn.Module):
    def __init__(self, num_emb, emb_dim, B = 0.25):
        super().__init__()
        # Note that num_emb = K, emb_dim = D
        # B is the commitment loss assuring that the encoder commits to an embedding (p. 4)
        self.emb_dim = emb_dim
        self.num_emb = num_emb

        # Initialize an embedding table and fill it with uniform values with expected mean = 0
        self.embedding = nn.Embedding(num_emb, emb_dim) # codebook of size [K, D]
        self.embedding.weight.data.uniform_(-1/self.num_emb, 1/self.num_emb)
        self.B = B

    def forward(self, z_e):
        # Take the input z_e from the encoder and add it to the embedding
        b,c,h,w = z_e.shape
        z_bhwc = z_e.permute(0,2,3,1) # Reformat by moving channel to the end

        # Flatten the z into a [b*h*w, emb_dim] vector
        z_flattened = z_bhwc.reshape(b*h*w, self.emb_dim)
        # Note: e = self.embedding.weight

        # Calculate the distance using ||a-b||
        dist = (torch.sum(z_flattened**2, -1, keepdim=True) +
               torch.sum(self.embedding.weight.t()**2, dim=0,keepdim=True) -
               2*torch.matmul(z_flattened, self.embedding.weight.t())
        )
        # We want to minimize the loss (page 3 in the paper)
        enc_idx = dist.argmin(dim = -1)

        # out shape : [b*h*w, emb_dim] (idx in the embedding)
        z_q = self.embedding(enc_idx)
        z_q = z_q.reshape(b,h,w,self.emb_dim)
        # Going from [b,h,w,emb_dim]->[b,emb_dim,h,w] to match input
        z_q = z_q.permute(0, 3, 1, 2)

        # Trick to counteract that the embedding is non-differentiable
        # This is the L on page 4 (TODO: explain these steps better!)
        reconstruction_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = self.B*F.mse_loss(z_q.detach(), z_e)
        loss = reconstruction_loss + commitment_loss

        z_q = z_e + (z_q - z_e).detach()

        # z_q shape is [b,emb_dim,h,w]
        return z_q, loss, enc_idx
    
class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.residual = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1),
        )

    def forward(self, x):
        return x + self.residual(x)
    

class VQEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels = 256):
        # We need to form a [ze_(x),D] object (see p. 4)
        # Inspired by p. 5
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels,
                               kernel_size = 4, stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels,
                               kernel_size = 4, stride = 2, padding = 1)

        self.activation = nn.ReLU()

        # We create two residual blocks as in the paper (p. 5)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels, ),
            ResidualBlock(hidden_channels)
        ])

    def forward(self, x):
        # Encoder consisits of 2 convolutional layer followed by residual blocks
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        for f in self.res_blocks:
            x = f(x)
        # [B, hidden_channels, H, W]
        return x

class VQDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Use in_channels = emb_dim

        self.res_blocks = nn.ModuleList([
            ResidualBlock(in_channels),
            ResidualBlock(in_channels)
        ])
        # Small change

        self.deconv1 = nn.ConvTranspose2d(in_channels, in_channels,
                                          kernel_size=4, stride = 2, padding = 1)
        self.deconv2 = nn.ConvTranspose2d(in_channels, out_channels,
                                          kernel_size=4, stride = 2, padding = 1)

        self.activation1 = nn.ReLU()
        self.activation2 = nn.Identity()

    def forward(self, z):
        for f in self.res_blocks:
            z = f(z)
        x = self.deconv1(z)
        x  = self.activation1(x)
        x = self.deconv2(x)
        x = self.activation2(x)
        return x
    
class VQVAE(nn.Module):
    def __init__(self, in_channels, hidden_dim, K, B = 0.4):
        super().__init__()
        self.encoder = VQEncoder(in_channels, hidden_dim)
        self.embedding = VQEmbedding(K, hidden_dim, B = B)
        self.decoder = VQDecoder(hidden_dim, in_channels)

    def forward(self, x):
        z = self.encoder(x)
        z_q, emb_loss, enc_idx = self.embedding(z)
        x_hat = self.decoder(z_q)

        return x_hat, emb_loss, enc_idx
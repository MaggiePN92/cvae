import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, mask, dilation = 1, **kwargs):
        # We want to mask the convolution such that it only views left or top values
        # Dilation reduces the active convolution cells

        super().__init__()
        kernel_size = mask.shape
        # Fancy formula to ensure that padding is correct
        padding = tuple([dilation*(kernel_size[i]-1)//2 for i in range(2)])

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding = padding, dilation = dilation, **kwargs)

        # Initiate the mask for the convolution of shape [1,1,*kernel_size]
        self.register_buffer("mask", mask[None, None])

    def forward(self, x):
        masked_weights = self.conv.weight * self.mask
        return F.conv2d(x, masked_weights, self.conv.bias,
                        stride=self.conv.stride,
                        padding=self.conv.padding,
                        dilation=self.conv.dilation,
                        groups=self.conv.groups)
    
class VerticalStackConvolution(MaskedConvolution):
    def __init__(self, in_channels, out_channels, kernel_size = 3, mask_center = False, dilation = 1, **kwargs):
        # We want to zero out all values below the middle in the convolutional kernel
        assert kernel_size % 2 == 1
        center = kernel_size // 2

        mask = torch.ones(kernel_size, kernel_size)
        mask[center + 1:, :] = 0

        # For the first convolution we also want to mask the center
        # This because it otherwise knows about the state it should learn
        if mask_center:
            mask[center, :] = 0

        super().__init__(in_channels, out_channels, mask, dilation, **kwargs)

class HorizontalStackConvolution(MaskedConvolution):
    def __init__(self, in_channels, out_channels, kernel_size = 3, mask_center = False, dilation = 1, **kwargs):
        assert kernel_size % 2 == 1
        center = kernel_size // 2
        # We want to zero out the row to the right of the middle in the convolutional kernel
        mask = torch.ones(1, kernel_size)
        # mask[:kernel_size // 2 + 1, kernel_size // 2 + 1] if we would use the full kernel here
        mask[0, center + 1:] = 0

        # For the first convolution we also want to mask the center
        if mask_center:
            mask[0, center] = 0

        super().__init__(in_channels, out_channels, mask, dilation, **kwargs)

class GatedMaskedConv(nn.Module):
    def __init__(self, in_channels, w_dim, dilation=1, kernel_size = 3):
        super().__init__()
        # We use two channels visualized as 2p in the paper (p. 4)
        self.conv_vstack = VerticalStackConvolution(in_channels, 2 * in_channels, kernel_size, dilation = dilation)
        self.conv_hstack = HorizontalStackConvolution(in_channels, 2 * in_channels, kernel_size, dilation = dilation)
        # We use the self.conv_transform to learn a channel representation that effectively combines the whole kernel
        self.conv_vh = nn.Conv2d(2 * in_channels, 2 * in_channels, kernel_size = 1)
        self.conv_hh = nn.Conv2d(in_channels, in_channels, kernel_size = 1)

        # Linearly project the class projections
        self.linear_v = nn.Linear(w_dim, 2*in_channels)
        self.linear_h = nn.Linear(w_dim, 2*in_channels)

    def forward(self, v_stack, h_stack, w):
        # Getting conditional bias, shape [B, w_dim] -> [B, 2*in_channels, 1, 1]
        v_bias = self.linear_v(w).unsqueeze(-1).unsqueeze(-1)
        h_bias = self.linear_h(w).unsqueeze(-1).unsqueeze(-1)

        # The vertical stack (see fig. 1 on p. 2)
        v = self.conv_vstack(v_stack) + v_bias # [B, 2*in_channels, H, W]
        # In fig. 2 (p. 4) marked as tanh and sigmoid through p (left)
        v_val, v_gate = v.chunk(2, dim = 1) # Split the result into chunks
        v_out = torch.tanh(v_val) * torch.sigmoid(v_gate)

        # The horizontal stack
        h = self.conv_hstack(h_stack) + h_bias
        h += self.conv_vh(v) # Add the information from the vertical stack

        # In fig. 2 (p. 4) marked as tanh and sigmoid through p (right)
        h_val, h_gate = h.chunk(2, dim = 1)
        h = torch.tanh(h_val) * torch.sigmoid(h_gate)
        h_out = self.conv_hh(h)
        # Add back the original value
        h_out += h_stack

        return v_out, h_out
    
class PixelCNN(nn.Module):
    def __init__(self, K, hidden_dimension, emb_dim, num_scores = 10, kernel_size = 3):
        """_summary_

        Args:
            in_dimension (_type_): Describes the channel size or embedding dimension
            hidden_dimension (_type_): Describes the dimension that we want to channel to during the convolution
            num_classes (_type_): Number of classes in the training
            emb_dim (_type_): _description_
            kernel_size (int, optional): _description_. Defaults to 3.
            K (int, optional): _description_. Defaults to 0.
            mode (str, optional): _description_. Defaults to "pixel".
        """
        assert emb_dim % 2 == 0
        super().__init__()
        # We assume that in_dimension = out_dimension
        self.num_scores = num_scores

        self.code_emb = nn.Embedding(K, hidden_dimension)
        # self.class_projection = nn.Linear(num_scores, emb_dim) # Add projection for class predictions if continous
        self.score_emb0 = nn.Embedding(num_scores, emb_dim // 2) # Add embedding for class predictions if discrete
        self.score_emb1 = nn.Embedding(num_scores, emb_dim // 2) # Add embedding for class predictions if discrete

        # First convolution skips center pixel as this has not been observed
        self.conv_vstack = VerticalStackConvolution(hidden_dimension, hidden_dimension, kernel_size, mask_center = True)
        self.conv_hstack = HorizontalStackConvolution(hidden_dimension, hidden_dimension, kernel_size, mask_center = True)

        # Repeated convolution with changing dilation
        self.conv_steps = nn.ModuleList([
            GatedMaskedConv(hidden_dimension, emb_dim),
            GatedMaskedConv(hidden_dimension, emb_dim, dilation = 2),
            GatedMaskedConv(hidden_dimension, emb_dim),
            GatedMaskedConv(hidden_dimension, emb_dim, dilation = 4),
            GatedMaskedConv(hidden_dimension, emb_dim),
            GatedMaskedConv(hidden_dimension, emb_dim, dilation = 2),
            GatedMaskedConv(hidden_dimension, emb_dim)
        ])

        self.conv_out = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(hidden_dimension, K, kernel_size = 1)
        )

    def forward(self, z, class_weights):
        # Assume z is a grid containing values to the codebook with K-rows (same as VQ-VAE)
        # class_weights has to have shape [B, num_classes]
        assert class_weights.shape[1] == 2 # We are now using two classes (Moira and Fernando)
        assert class_weights.dim() == 2 

        z = self.code_emb(z)
        z = z.permute(0, 3, 1, 2).contiguous() # [B, H_l, W_l, h_dim] -> [B, h_dim, H_l, W_l]

        ids = class_weights.to(dtype = torch.long)
        e0 = self.score_emb0(ids[:, 0])  # -> [B, emb_dim//2]
        e1 = self.score_emb1(ids[:, 1])  # -> [B, emb_dim//2]
        cond = torch.cat([e0, e1], dim = 1) # -> [B, emb_dim]
        v_stack = self.conv_vstack(z)
        h_stack = self.conv_hstack(z)

        # Perform convolution on both stacks at the same time
        for conv in self.conv_steps:
            v_stack, h_stack = conv(v_stack, h_stack, cond)

        logits = self.conv_out(h_stack)

        return logits

    def negative_likelihood(self, x, class_weights):
        logits = self.forward(x, class_weights)
        neg_ll = F.cross_entropy(logits, x, reduction="mean")
        return neg_ll

    def sample_prior(self, class_weights, shape, device, temperature=1.0):
        # This function is modified for vq-vae
        B = class_weights.size(0)
        H_l, W_l = shape # Shape of   [H_l, W_l]

        # The encoded shape is a grid of indexes of shape [B, H_l, W_l]
        index_grid = torch.zeros(B, H_l, W_l, dtype=torch.long, device=device)

        for h in range(H_l):
            for w in range(W_l):
                logits = self.forward(index_grid, class_weights)    # (B,K,H_l,W_l)
                pixel_logits = logits[:, :, h, w] / temperature    # (B,K)
                probs = pixel_logits.softmax(dim=1)
                sample = torch.multinomial(probs, 1).squeeze(1)    # (B,)
                index_grid[:, h, w] = sample
            print("Height:", h)
        return index_grid
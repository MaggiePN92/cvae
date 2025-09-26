import torch
from torch.optim import Adam
from tqdm import tqdm
from .vae_loss import vae_loss


def train_loop(model, train_loader, device, epochs=10, lr=1e-4):
    # adam optimizer
    optimizer = Adam(model.parameters(), lr=lr)
    
    model.train()
    
    for epoch in range(epochs):
        train_bar = tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}")
        
        for batch_idx, (x, c) in enumerate(train_bar):
            x = x.to(device)
            c = c.to(device)
            
            # forward pass
            x_hat, mean, logvar, c_out = model(x, c)
            
            # beta schedule - start low, increase over time
            beta = min(1.0, (epoch + 1) / 30.0)
            
            # compute loss
            loss, recon, kl = vae_loss(x, x_hat, mean, logvar, beta)
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # update progress bar
            train_bar.set_postfix({
                "loss": f"{loss.item():.4e}",
                "recon": f"{recon.item():.4e}",
                "kl": f"{kl.item():.4e}",
                "beta": f"{beta:.3f}"
            })
    
    return model

from .ae_loss import ae_loss
import torch
import torch.nn as nn
from tqdm import tqdm


def train_ae_epoch(
    ae,
    loader,
    opt,
    device,
    lam=1e-3,
    n_classes=6,
    epoch_idx=None,
    num_epochs=None,
    log_grad_norm=False,
):
    ae.train()
    tot = tot_recon = tot_reg = 0.0
    n_samples = 0

    desc = f"epoch {epoch_idx+1}/{num_epochs}" if (epoch_idx is not None and num_epochs is not None) else "train"
    train_bar = tqdm(loader, desc=desc)

    for batch_idx, (x, c) in enumerate(train_bar):
        x = x.to(device)
        c = c.to(device).reshape(x.size(0), n_classes)

        # forward
        c_vec = ae.scale_class_vec(c) # [B, n_classes]
        z = ae.encode(x, c_vec) # [B, z_dim]
        x_hat = ae.decode(z, c_vec) # [B, C, H, W]

        # basic guards
        assert x_hat.shape == x.shape, f"pred {tuple(x_hat.shape)} vs target {tuple(x.shape)}"
        if torch.isnan(z).any() or torch.isinf(z).any():
            raise RuntimeError("NaN/Inf detected in latent z")
        if torch.isnan(x_hat).any() or torch.isinf(x_hat).any():
            raise RuntimeError("NaN/Inf detected in x_hat")

        # loss
        loss, recon, reg = ae_loss(x, x_hat, z, lam)

        # backward
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if log_grad_norm:
            total_grad_norm = nn.utils.clip_grad_norm_(ae.parameters(), 1.0).item()
        else:
            nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            total_grad_norm = None
        opt.step()

        # running sums
        bs = x.size(0)
        n_samples += bs
        tot += loss.item() * bs
        tot_recon += recon.item() * bs
        tot_reg += reg.item() * bs

        # tqdm postfix (instant values, not epoch avg)
        postfix = {
            "loss": f"{loss.item():.4e}",
            "recon": f"{recon.item():.4e}",
            "reg": f"{reg.item():.4e}",
            "λ": f"{lam:.1e}",
        }
        if log_grad_norm and total_grad_norm is not None:
            postfix["|g|"] = f"{total_grad_norm:.3f}"
        train_bar.set_postfix(postfix)

    # epoch averages
    avg_loss = tot / n_samples
    avg_recon = tot_recon / n_samples
    avg_reg = tot_reg / n_samples
    return avg_loss, avg_recon, avg_reg

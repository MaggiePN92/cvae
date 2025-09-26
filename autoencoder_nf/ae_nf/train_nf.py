import torch
from tqdm import tqdm
from .latent_flow import LatentFlow

@torch.no_grad()
def encode_batch(ae, x, c):
    ae.eval()
    c_vec = ae.scale_class_vec(c)
    z = ae.encode(x, c_vec)
    return z, c_vec


def train_flow_epoch(
    flow : LatentFlow,
    ae,
    loader,
    opt,
    device,
    n_classes=6,
    epoch_idx=None,
    num_epochs=None,
    log_grad_norm=False,
):
    flow.train() 
    ae.eval()
    tot = 0.0
    n_samples = 0

    desc = f"epoch {epoch_idx+1}/{num_epochs}" if (epoch_idx is not None and num_epochs is not None) else "train_flow"
    train_bar = tqdm(loader, desc=desc)

    for batch_idx, (x, c) in enumerate(train_bar):
        x = x.to(device)
        c = c.to(device).reshape(x.size(0), n_classes)

        with torch.no_grad():
            # retrieve latent space and prefrence vector 
            z, c_vec = encode_batch(ae, x, c)

        if not torch.isfinite(z).all():
            raise RuntimeError("NaN/Inf in z from AE during flow training")

        # negative log likelihood
        # adapting flow to latent space 
        nll = -flow.log_prob(z, cond=c_vec).mean()

        if not torch.isfinite(nll):
            raise RuntimeError("NaN/Inf in flow NLL")

        opt.zero_grad(set_to_none=True)
        nll.backward()

        # clip grads to avoid exploiding gradients 
        if log_grad_norm:
            total_grad_norm = torch.nn.utils.clip_grad_norm_(
                flow.parameters(), 1.0).item()
        else:
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
            total_grad_norm = None

        opt.step()

        bs = x.size(0)
        n_samples += bs
        tot += nll.item() * bs

        postfix = {
            "nll": f"{nll.item():.4e}"
        }
        if log_grad_norm and total_grad_norm is not None:
            postfix["|g|"] = f"{total_grad_norm:.3f}"
        train_bar.set_postfix(postfix)

    avg_nll = tot / n_samples
    return avg_nll

from cvae.cvae import CVAE
from data.read_data import get_cars_dataloader
import torch 
from utils.generate_img import generate_img
from torch.optim import Adam
from tqdm import tqdm
from training.vae_loss import vae_loss
from torch.utils.data import Subset
from cvae.latent_flow import LatentFlow


def train_loop(
        conditioned_model,
        train_loader,
        device,
        opt_alg=Adam,
        epochs=10,
        lr=3e-4,
        n_classes=6
    ):
    cvae = conditioned_model.to(device)
    optimizer = opt_alg(cvae.parameters(), lr=lr)
    cvae.train()

    for epoch in range(epochs):
        train_bar = tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}")
        for batch_idx, (x, c) in enumerate(train_bar):
            x = x.to(device)
            # ensure shape [B, n_classes]; never squeeze away batch dim
            c = c.to(device).reshape(x.size(0), n_classes)

            x_hat, mean, logvar = cvae(x, c)
            
            assert x_hat.shape == x.shape,\
                f"pred {tuple(x_hat.shape)} vs target {tuple(x.shape)}"

            # guard against NaNs in encoder outputs
            if torch.isnan(mean).any() or torch.isnan(logvar).any():
                raise RuntimeError("NaN detected in mean/logvar")
            logvar = torch.clamp(logvar, -10.0, 10.0)

            # beta schedule: 0->1 over 30 epochs
            beta = min(1.0, (epoch + 1) / 30.0)
            # beta = 0 

            loss, recon, kl = vae_loss(x, x_hat, mean, logvar, beta)
            # loss = recon
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cvae.parameters(), 1.0)
            optimizer.step()

            train_bar.set_postfix({
                "loss": f"{loss.item():.4e}",
                "recon": f"{recon.item():.4e}",
                "kl": f"{kl.item():.4e}",
                "beta": f"{beta:.3f}"
            })

    return cvae


def main(batch_size=8, epochs=100, lr=3e-4, test = True, test_samples=100):
    train_loader = get_cars_dataloader(batch_size=batch_size)
    
    if test:
        dataset = train_loader.dataset
        epochs = 5
        hidden_flow = 12
        k = 2
        small_idx = list(range(test_samples))
        small_ds = Subset(dataset, small_idx)
        train_loader = torch.utils.data.DataLoader(
            small_ds, batch_size=32, shuffle=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Device set as:", device)

    z_dim = 8
    n_classes = 6
    conditioned_model = CVAE(
        z_dim=z_dim,
        n_classes=n_classes,
        n_channels=3,
        img_size=[96, 128]
    ).to(device)

    trained_cvae = train_loop(
        conditioned_model=conditioned_model,
        train_loader=train_loader,
        device=device,
        epochs=epochs,
        lr=lr,
        n_classes=n_classes
    )

    flow = LatentFlow(
        z_dim, cond_dim=6, hidden=hidden_flow, K=k).to(device)
    for epoch in range(epochs):
        flow_loss = train_flow(flow)


def train_flow():
    pass 

if __name__ == "__main__":
    main()

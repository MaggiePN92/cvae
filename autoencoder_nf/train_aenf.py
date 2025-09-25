import torch
from ae_nf.cond_AE import CondAE
from ae_nf.train_ae import train_ae_epoch
from ae_nf.latent_flow import LatentFlow
from ae_nf.train_nf import train_flow_epoch
from ae_nf.sample_imgs import sample_images
from data.read_data import get_cars_dataloader
from utils.gen_imgs_aenf import save_generated_images

test = True
n_samples = 100
n_epochs = 3
z_dim = 8
batch_size = 32
train_loader = get_cars_dataloader(batch_size=batch_size)

if test:
    from torch.utils.data import Subset

    dataset = train_loader.dataset
    small_idx = list(range(n_samples))
    small_ds = Subset(dataset, small_idx)
    train_loader = torch.utils.data.DataLoader(small_ds, batch_size=32, shuffle=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device set as:", device)

# stage 1: Training the Auto Encoder - creating a latent space to sample from
ae = CondAE(z_dim=z_dim, n_classes=6, n_channels=3, img_size=(96,128)).to(device)
opt_ae = torch.optim.Adam(ae.parameters(), lr=3e-4)
for epoch in range(n_epochs):
    loss, recon, reg = train_ae_epoch(
        ae, train_loader, opt_ae, device, lam=1e-3,
        num_epochs=n_epochs, epoch_idx=epoch)

# stage 2: Training the latent flow 
flow = LatentFlow(dim=z_dim, cond_dim=6, hidden=128, K=4).to(device)
opt_flow = torch.optim.Adam(flow.parameters(), lr=1e-3)
for epoch in range(n_epochs):
    nll = train_flow_epoch(
        flow, ae, train_loader, opt_flow, device,
        num_epochs=n_epochs, epoch_idx=epoch)

# sample and generate images 
c_raw = torch.tensor([[1,9,1,1,1,1]], dtype=torch.float32, device=device)
x_gen = sample_images(ae, flow, c_raw, num=5, device=device)
save_generated_images(x_gen, out_dir="aenf_output", num_images=5)

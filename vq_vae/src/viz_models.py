import torch
from src.PixelCNN import PixelCNN
from src.VQVAE import VQVAE
from src.train_model import load_cars
from src.utils import get_loss_e, load_model, set_seed, unnormalize_tensor
from src.viz_utils import plot_n, plot_n_pairs, visualize_epochs

def visualize_vqvae(vq_model, x, device, n=4):
    # limit n to batch size
    B = x.size(0)
    n = min(n, B)

    # run through model (move inputs to device used by the script)
    with torch.no_grad():
        x_dev = x.to(device)
        z = vq_model.encoder(x_dev)
        z_q, _, enc_idx = vq_model.embedding(z)
        recon = vq_model.decoder(z_q)

    # unnormalize both original and reconstruction and bring to cpu
    x_unnorm = unnormalize_tensor(x_dev).cpu()
    recon_unnorm = unnormalize_tensor(recon).cpu()

    print("recon min/max/mean:", float(recon_unnorm.min()), float(recon_unnorm.max()), float(recon_unnorm.mean()))
    plot_n_pairs(x_unnorm, recon_unnorm, n)

def visualize_combined(vq_model, prior, device, class_weights, emb_size, temp):
    # Code written with GPT-5
    vq_model.eval()
    prior.eval()
    with torch.no_grad():
        H_l, W_l = emb_size

        index_grid = prior.sample_prior(class_weights, (H_l, W_l), device, temperature = temp)  # [n, H_l, W_l]

        z_q = vq_model.embedding.embedding(index_grid).permute(0, 3, 1, 2).contiguous()          # [n, D, H_l, W_l]
        x_hat = vq_model.decoder(z_q).cpu() # [n, 3, 32, 32]               
        
        x_hat = unnormalize_tensor(x_hat)

    x = []
    x_gen = []
    for i in range(x_hat.size(0)):
        x.append(torch.zeros_like(x_hat[i]))
        x_gen.append(x_hat[i])
    
    n = len(x)
    plot_n(x_gen, n = n)


if __name__ == "__main__":
    set_seed(27) # For reproducibility

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- VQVAE ---
    vq_model = VQVAE(
        in_channels=3, hidden_dim=256, K = 512
        ).to(device)

    vqvae_info = load_model(
        vq_model, "output/model/vavae.pt", map_location = device, strict = True
        )
    print(vqvae_info["config"]) # Check that the initialized model is the same
    vq_epochs, vq_losses, vq_durations = get_loss_e(vqvae_info["extra"])
    visualize_epochs(vq_losses, vq_epochs)

    # --- PIXELCNN ---
    pixel_model = PixelCNN(
        K = 512, hidden_dimension = 256, num_scores = 10, emb_dim = 64, kernel_size = 3
        ).to(device)

    pixel_info = load_model(pixel_model, "output/model/pixelcnn.pt", map_location = device, strict = True)
    print(pixel_info["config"]) # Check that the initialized model is the same
    pixel_epochs, pixel_losses, pixel_durations = get_loss_e(pixel_info["extra"])
    visualize_epochs(pixel_losses, pixel_epochs)
    
    n = 10
    class_weights = torch.tensor([[2, 8]] * n, device=device, dtype=torch.long)
    visualize_combined(vq_model, pixel_model, device, class_weights = class_weights, emb_size = [24, 32], temp = 0.5)

    # --- VQVAE visualized ---    # Load normalized training data
    in_mean = [0.485, 0.456, 0.406]
    in_std = [0.229, 0.224, 0.225]

    data_path = "./data"
    train_loader = load_cars(data_path, in_mean, in_std)

    #n = 8
    #x_batch, c_batch = next(iter(train_loader))
    #x_n = x_batch[:n]

    #visualize_vqvae(vq_model, x_n, device, n = 8)
    #visualize_distribution(train_loader)






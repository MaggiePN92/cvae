import numpy as np
import torch
from tqdm import tqdm
from src.PixelCNN import PixelCNN
from src.VQVAE import VQVAE
from extra.dataloader import load_cifar
from src.utils import get_loss_e, load_model, preprocess_weights

def run_loop(vq_model, prior_model, image_loader, device, joint_class = [1,3]):
    # Freeze the vq_model
    vq_model.eval()
    for p in vq_model.parameters():
        p.requires_grad = False

    prior_model.train()

    for x, c in image_loader:
        x = x.to(device, non_blocking = True)
        c = c.to(device, non_blocking = True)
        c = preprocess_weights(c, idx = joint_class)

        # Returns the embedding that is used in the prior
        #idx_map = get_idxmap(vq_model, x)
        break

def decode_indices_to_images(vqvae, index_grid, device):
    # index_grid: [B,H_l,W_l] long
    vqvae = vqvae.to(device).eval()
    with torch.no_grad():
        # Adapt: if your vqvae has a decoder that accepts indices, use it.
        # Example assumes vqvae.decode_from_indices returning [B,3,H,W] float in normalized space
        imgs = vqvae.decode_from_indices(index_grid.to(device))
        return imgs.cpu()

def sample_and_diagnose(pixelcnn, vqvae, class_weights, shape, device, n_samples=8, temperature=1.0):
    pixelcnn = pixelcnn.to(device).eval()
    class_weights = class_weights.to(device).float()
    samples = []
    uniques = []
    with torch.no_grad():
        for i in range(n_samples):
            idx_grid = pixelcnn.sample_prior(class_weights, shape, device, temperature=temperature)  # [B,H,W]
            samples.append(idx_grid.cpu().numpy())
            uniques.append(len(np.unique(idx_grid.cpu().numpy())))
    print("Unique codes per sample:", uniques)
    # decode first batch example if vqvae.decode exists
    decoded = decode_indices_to_images(vqvae, samples[0][0], device)  # adapt API
    # show images using matplotlib here or save
    return {"samples": samples, "uniques": uniques, "decoded_example": decoded}

def test_masking_invariance(pixelcnn, device=None, K=512):
    device = device or next(pixelcnn.parameters()).device
    pixelcnn = pixelcnn.to(device).eval()

    B, H, W = 1, 8, 8
    # class_weights dummy (must match num_scores)
    class_weights = torch.zeros(1, pixelcnn.num_scores, device=device, dtype=torch.float32)

    # target pixel (row, col)
    h_target, w_target = 2, 3

    base = torch.zeros(B, H, W, dtype=torch.long, device=device)
    a = base.clone()
    b = base.clone()

    # Change a future position in 'a' and 'b' differently:
    # future = positions (r,c) where r > h_target or (r == h_target and c > w_target)
    # pick (h_target+1, 0) which is strictly future
    a[0, h_target+1, 0] = 5
    b[0, h_target+1, 0] = 7

    with torch.no_grad():
        logits_a = pixelcnn.forward(a, class_weights)  # [B,K,H,W]
        logits_b = pixelcnn.forward(b, class_weights)

    # Compare logits at target pixel
    la = logits_a[:, :, h_target, w_target].cpu().numpy()
    lb = logits_b[:, :, h_target, w_target].cpu().numpy()
    diff = np.abs(la - lb).max()
    print("max diff at target pixel logits:", diff)
    assert diff < 1e-6, "Masking violated: future pixels changed current pixel logits"
    print("Masking test passed.")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- VQVAE ---
    vq_model = VQVAE(in_channels=3, hidden_dim=256, K = 512)

    vqvae_info = load_model(
        vq_model, "model/vavae.pt", map_location = device, strict = True
        )
    print(vqvae_info["config"]) # Check that the initialized model is the same
    vq_epochs, vq_losses, vq_durations = get_loss_e(vqvae_info["extra"])
    #visualize_epochs(vq_losses, vq_epochs)

    # --- PIXELCNN ---
    pixel_model = PixelCNN(K = 512, hidden_dimension = 256, num_scores = 1, emb_dim = 64, kernel_size = 3)

    pixel_info = load_model(
        pixel_model, "model/pixelcnn.pt", map_location = device, strict = True
        )
    print(pixel_info["config"]) # Check that the initialized model is the same
    pixel_epochs, pixel_losses, pixel_durations = get_loss_e(pixel_info["extra"])

    # --- DATA ---
    in_mean = [0.485, 0.456, 0.406]
    in_std = [0.229, 0.224, 0.225]

    data_path = "."
    image_loader = load_cifar(data_path)

    #run_loop(vq_model, pixel_model, image_loader, device)
    sample_and_diagnose(pixel_model, vq_model, class_weights = torch.full((1,1), fill_value = 0.5, device=device, dtype=torch.float), 
                        shape = [24,32], device = device, n_samples=1)



import torch
import torch.nn.functional as F
from src.ModelTracker import ModelTracker
import quixdata as quix
import torchvision.transforms as T
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import classes and functions
from src.utils import get_idxmap, load_model, preprocess_weights, save_model, set_seed
from src.VQVAE import VQVAE
from src.PixelCNN import PixelCNN


def train_vqvae(vq_model, optimizer, train_loader, device, tracker, epochs = 15):
    vq_model.train()
    tracker.reset()

    for epoch in range(epochs):
        total_loss = 0.0
        total_examples = 0

        train_bar = tqdm(iterable = train_loader)
        for x, c in train_bar:
            x = x.to(device, non_blocking = True)

            optimizer.zero_grad(set_to_none=True)
            x_hat, emb_loss, enc_idx = vq_model(x)

            reconstruction_loss = F.mse_loss(x_hat, x, reduction = "mean")
            loss = reconstruction_loss + emb_loss

            loss.backward()
            optimizer.step()

            b_sz = x.size(0) # batch size
            total_loss += loss.item() * b_sz
            total_examples += b_sz

            train_bar.set_description(f'Epoch [{epoch+1}/{epochs}]')
            train_bar.set_postfix(
                emb_loss = f"{emb_loss.item():.3f}",
                total_loss = f"{loss.item():.3f}"
                )
        avg_total = total_loss / total_examples
        tracker.update(avg_total, epoch)

def train_combined(vq_model, prior_model, optimizer, scheduler, train_loader, device, tracker, joint_class = [1,3], epochs = 15):
    # Freeze the vq_model
    vq_model.eval()
    for p in vq_model.parameters():
        p.requires_grad = False

    prior_model.train()
    tracker.reset()

    for epoch in range(epochs):
        total_loss = 0.0
        total_examples = 0

        train_bar = tqdm(iterable = train_loader)
        train_bar.set_description(f'Epoch [{epoch+1}/{epochs}]')

        for x, c in train_bar:
            x = x.to(device, non_blocking = True)
            c = c.to(device, non_blocking = True, dtype = torch.long)
            c = preprocess_weights(c, idxs = joint_class)
            
            optimizer.zero_grad(set_to_none=True)

            # Returns the embedding that is used in the prior
            idx_map = get_idxmap(vq_model, x)

            # Use the indices for training in the pixelcnn conditioned on c
            loss = prior_model.negative_likelihood(idx_map, c)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prior_model.parameters(), max_norm=1.3) # Add clipping for regularization
            optimizer.step()

            b_sz = x.size(0)
            total_loss += loss.item() * b_sz
            total_examples += b_sz

            train_bar.set_postfix(
                loss = f"{loss.item():.3f}"
                )
        avg_loss = total_loss / total_examples
        tracker.update(avg_loss, epoch)
        scheduler.step() # Add a scheduler that adjust lr 

def load_cars(data_path, in_mean, in_std, num_workers = 0):
    postprocess = (T.Compose([T.ToTensor(), T.Normalize(in_mean, in_std)]), 
                T.ToTensor())

    data = quix.QuixDataset(
        "CarRecs", 
        data_path, 
        override_extensions = [
            'jpg',
            'scores.npy'
        ]
    ).map_tuple(*postprocess)

    #with traindata.shufflecontext():
    N = len(data)

    # Set a suitable batch size
    batch_size = 128

    # Create data loader
    train_loader = DataLoader(data, shuffle=True, batch_size=batch_size, num_workers = num_workers)

    # names = ["Raof", "Moira", "Louie", "Ferdinando", "Gragar", "Esther"]
    return train_loader

if __name__ == "__main__":
    set_seed(27) # For reproducibility

    # Set the main device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load normalized training data
    in_mean = [0.485, 0.456, 0.406]
    in_std = [0.229, 0.224, 0.225]

    data_path = "/projects/ec517/data/"
    train_loader = load_cars(data_path, in_mean, in_std)

    hidden_dim = 256
    K = 512
    lr = 3e-4
    num_scores = 10 # We have ten different scores per person
    epochs_vq = 100
    epochs_prior = 200
    class_emb_dim = 64
    weight_decay = 1e-6
    
    # ---- VQ-VAE ----
    vq_tracker = ModelTracker()
    vq_model = VQVAE(
        in_channels = 3, # rgb
        hidden_dim = hidden_dim, 
        K = K
        ).to(device)
    vq_optimizer = Adam(vq_model.parameters(), lr = lr)

    vq_model_path = "./model/vavae.pt"
    pre_trained_vq = True
    if pre_trained_vq: 
        load_model(
            vq_model, vq_model_path, map_location = device, strict = True
        )
    else:
        train_vqvae(vq_model, vq_optimizer, 
                    train_loader, device, vq_tracker, epochs = epochs_vq)

        vq_config = {"K": K, "hidden_dim": hidden_dim}
        vq_extra = vq_tracker.to_dict()
        save_model(vq_model, vq_model_path, 
                name = "VQVAE", config = vq_config, extra = vq_extra)

    # ---- PixelCNN Prior ----
    prior_tracker = ModelTracker()
    pixelcnn_model = PixelCNN(
        K=K, # 512
        hidden_dimension = hidden_dim, # 256
        num_scores = num_scores, # Not used in current implementation
        emb_dim = class_emb_dim, # 64
    ).to(device)
    prior_optimizer = AdamW(pixelcnn_model.parameters(), lr = lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(prior_optimizer, step_size=60, gamma=0.7)

    pixelcnn_model_path = "./model/pixelcnn.pt"
    pre_trained_pixel = False
    if pre_trained_pixel: 
        load_model(
            pixelcnn_model, pixelcnn_model_path, map_location = device, strict = True
        )
    else:
        train_combined(vq_model, pixelcnn_model, 
                    prior_optimizer, scheduler, train_loader, device, prior_tracker, epochs = epochs_prior)

        prior_config = {"K": K, "hidden_dim": hidden_dim, "class_emb_dim": class_emb_dim}
        prior_extra = prior_tracker.to_dict()
        save_model(pixelcnn_model, pixelcnn_model_path, 
                name = "PIXELCNN", config = prior_config, extra = prior_extra)

    
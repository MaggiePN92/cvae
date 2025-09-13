from training.train_loop import train_loop
from cvae.cvae import CVAE
from data.read_data import get_cars_dataloader
import torch 
from utils.generate_img import generate_img

def main(batch_size = 128):
    data_loader = get_cars_dataloader(batch_size = batch_size)
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu")
    
    print(f"Device set as: {device}")
    z_dim = 2
    conditioned_model = CVAE(
        z_dim, 
        n_classes=6, 
        n_channels=3, 
        img_size=[96,128]
    ).to(device)

    trained_model = train_loop(
        conditioned_model, data_loader, device)
    
    generate_img(trained_model, z_dim, device, seed=42)

if __name__ == "__main__":
    main()

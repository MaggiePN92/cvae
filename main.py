import torch
from training.train_loop import train_loop
from cvae.cvae import CVAE
from data.read_data import get_cars_dataloader
from utils.generate_img import generate_img

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)
    

    data_loader = get_cars_dataloader(batch_size=64)
    
    model = CVAE(z_dim=32, condition_dim=2, n_channels=3, img_size=[96,128])
    model = model.to(device)

    trained_model = train_loop(model, data_loader, device, epochs=100)
    
    generate_img(trained_model, 32, device, seed=42)
    
    print("done!")

if __name__ == "__main__":
    main()
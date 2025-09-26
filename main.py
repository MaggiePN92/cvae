import torch
from training.train_loop import train_loop
from cvae.cvae import CVAE
from data.read_data import get_cars_dataloader
from utils.generate_img import generate_img
from torch.utils.data import Subset


def main(test = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)
    
    epochs = 100
    data_loader = get_cars_dataloader(batch_size=64)
    out_dir = "generated_imgs"
    z_dim = 32

    if test:
        epochs = 5
        z_dim = 2
        dataset = data_loader.dataset
        out_dir = "test_generated_imgs"
        small_idx = list(range(100))
        small_ds = Subset(dataset, small_idx)
        data_loader = torch.utils.data.DataLoader(small_ds, batch_size=32, shuffle=True) 
        
    model = CVAE(z_dim=z_dim, condition_dim=2, n_channels=3, img_size=[96,128])
    model = model.to(device)

    trained_model = train_loop(model, data_loader, device, epochs=epochs)
    
    generate_img(trained_model, z_dim, device, seed=42, out_dir=out_dir)
    
    print("done!")

if __name__ == "__main__":
    main(test = True)
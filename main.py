from train_loop import train_loop
from cvae.cvae import CVAE
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import torch 

def main():
    # Local folder where MNIST will be stored
    data_dir = "./data"

    # Define transform
    transform = transforms.ToTensor()

    # Download/train set
    train_set = MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_set = MNIST(root=data_dir, train=False, download=True, transform=transform)

    # Data loaders
    batch_size = 128
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device set as: {device}")
    z_dim = 2
    conditioned_model = CVAE(
        z_dim, n_classes=10, 
        n_channels=1, 
        img_size=[28,28]
    ).to(device)

    trained_model = train_loop(
        conditioned_model, train_loader, device)

if __name__ == "__main__":
    main()

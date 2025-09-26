from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

def load_cifar(data_dir = ".", batch_size = 128):

    # Get dataset
    transform = transforms.ToTensor()
    train_set = CIFAR10(root=data_dir, train=True, download=True, transform=transform)

    # Create data loader
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    return train_loader
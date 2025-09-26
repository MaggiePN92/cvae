
import quix # For quixdata public repo , use import quixdata as quix
import torchvision.transforms as T
from torch.utils.data import DataLoader


def get_cars_dataloader(datapath="./data/", batch_size=64):
    # Define mean and std from ImageNet data
    in_mean = [0.485 , 0.456 , 0.406]
    in_std = [0.229 , 0.224, 0.225]

    # Define postprocessing / transform of data modalities
    postprocess = ( # Create tuple for image and class ...
        T.Compose ([ # Handles processing of the .jpg image
            T.ToTensor (), # Convert from PIL image to torch.Tensor
            T.Normalize(in_mean , in_std), # Normalize image to correct mean/std.
        ]),
        T.ToTensor(), # Convert .scores.npy file to tensor.
    )

    # Load training data
    data = quix.QuixDataset(
        'CarRecs',
        datapath ,
        override_extensions =[ # Sets the order of the modalities:
            'jpg', # ... load image first ,
            'scores.npy' # ... load scores second.
        ],
    ).map_tuple (* postprocess)

    print("Data read:")
    print(data)

    with data.shufflecontext():
        data_loader = DataLoader(
            dataset=data, batch_size=batch_size)
    return data_loader


if __name__ == "__main__":
    dataloader = get_cars_dataloader()
    for x, a in dataloader:
        print(x.shape, a.shape)
        break
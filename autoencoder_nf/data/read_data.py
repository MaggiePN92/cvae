import quix
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch

def process_scores(scores):
    # just get moira and ferdinando scores
    scores = scores.squeeze()
    if scores.dim() == 1:
        scores = scores.unsqueeze(0)
    
    # moira is index 1, ferdinando is index 3
    moira_ferdinando = scores[:, [1, 3]]
    
    # normalize to 0-1
    moira_ferdinando = moira_ferdinando / 9.0
    
    if moira_ferdinando.dim() > 2:
        moira_ferdinando = moira_ferdinando.squeeze()
    
    return moira_ferdinando

def get_cars_dataloader(datapath="./data/", batch_size=64):
    # imagenet normalization
    in_mean = [0.485, 0.456, 0.406]
    in_std = [0.229, 0.224, 0.225]

    # transforms
    postprocess = (
        T.Compose([
            T.ToTensor(),
            T.Normalize(in_mean, in_std),
        ]),
        T.ToTensor(),
    )

    # load data
    data = quix.QuixDataset(
        'CarRecs',
        datapath,
        override_extensions=['jpg', 'scores.npy']
    ).map_tuple(*postprocess)

    print("data loaded:", data)

    # custom dataset wrapper
    class ProcessedDataset:
        def __init__(self, dataset):
            self.dataset = dataset
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            image, scores = self.dataset[idx]
            processed_scores = process_scores(scores)
            if processed_scores.dim() > 1:
                processed_scores = processed_scores.squeeze()
            return image, processed_scores

    processed_data = ProcessedDataset(data)

    with processed_data.dataset.shufflecontext():
        data_loader = DataLoader(dataset=processed_data, batch_size=batch_size)
    return data_loader
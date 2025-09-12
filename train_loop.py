import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
from vae_loss import vae_loss


def train_loop(
        conditioned_model, 
        train_loader, 
        device,
        opt_alg=Adam, 
        epochs=10, 
        lr=0.01,
    ):
    optimizer = opt_alg(conditioned_model.parameters(), lr = lr)

    # Train for a few epochs
    conditioned_model.train()

    for epoch in range(epochs):
        train_bar = tqdm(iterable=train_loader) # Progress bar
        
        for i, (x, c) in enumerate(train_bar):
            x = x.to(device)
            c = c.to(device) 
            # Get x_hat, mean, logvar,and cls_token from the conditioned_model
            x_hat, mean, logvar, cls_token = conditioned_model(x, c)

            # Get vae loss
            loss = vae_loss(x, x_hat, mean, logvar)

            # Get cross entropy loss for the cls token
            # cls_loss = F.cross_entropy(cls_token, F.one_hot(c, num_classes=10).double(), reduction='sum')
            cls_loss = F.cross_entropy(cls_token, c.long(), reduction='sum')
            # Add the losses as a weighted sum. NB: We weight the cls_loss by 10 here, but feel free to tweak it.
            loss = loss + cls_loss * 10
            
            # Update model parameters based on loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update progress bar
            train_bar.set_description(f'Epoch [{epoch+1}/{epochs}]')
            train_bar.set_postfix(loss = loss.item() / len(x))
    
    return conditioned_model

import matplotlib.pyplot as plt 
import torch
import torch.nn.functional as F


def generate_img(conditioned_model, z_dim, device):
    # Put model in evalulation mode
    conditioned_model.eval()

    # Select a class label
    cls_label = 7

    with torch.no_grad():
        # Sample noise from N(0,I)
        z = torch.randn(1, z_dim).to(device)

        # Make a one-hot for the selected class label, which will act as the cls token
        cls_token = F.one_hot(torch.tensor(cls_label).unsqueeze(0), num_classes=10).to(device)

        # Concatenate z and the cls token
        z = torch.cat((z, cls_token), dim=1)

        # Generate new image with the decoder
        x_hat = conditioned_model.decoder(z)
        x_hat = x_hat.squeeze(0).cpu().detach()

    # Show generated image
    plt.figure(figsize=(3,3))
    plt.imshow(x_hat.permute(1,2,0), cmap=plt.get_cmap('gray'))
    plt.title(f"A generated image of class={cls_label}")
    plt.axis('off')
    plt.show()

import torch
import torch.nn as nn
import os 
import torch
from src.cvae.cvae import CVAE


def load_trained_cvae(
    weights_path="output/final_model.pth",
    z_dim=32,
    condition_dim=2,
    n_channels=3,
    img_size=(96, 128),
    device=None,
    strict=True
):
    device = torch.device("mps" if torch.backends.mps.is_available() else
                      ("cuda" if torch.cuda.is_available() else "cpu"))
    model = CVAE(
        z_dim=z_dim,
        condition_dim=condition_dim,
        n_channels=n_channels,
        img_size=list(img_size),
    ).to(device)

    state = torch.load(weights_path, map_location=device)
    if strict:
        model.load_state_dict(state)
    else:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)

    model.eval()
    return model, device


def load_my_models() -> nn.Module:
    """
    Loads your model.
    Use the imports to load the framework for your model.
    Then load the state dict such that your model is loaded
    with the correct weights.
    """
    output_dir = "output"
    device = torch.device("mps" if torch.backends.mps.is_available() else
                      ("cuda" if torch.cuda.is_available() else "cpu"))
    full_path = os.path.join(output_dir, "final_model.pth")

    if os.path.exists(full_path):
        model, device = load_trained_cvae(
            weights_path="output/final_model.pth",
            device=device,
            z_dim=32
        )
        return model.decoder
    else:
        print(f"The path {full_path} doesn't exist")


def test_load_my_models():
    final_model = load_my_models()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_model = final_model.to(device)

    final_model.eval()

    # Send an example through the models, to check that they loaded properly
    images = torch.load('output/images.pth')
    embeddings = torch.load('output/embeddings.pth')["latent_codes"]
    cond_dim = getattr(final_model, "condition_dim", 2)
    B = embeddings.shape[0]
    c = torch.tensor([1.0, 1.0], device=device, dtype=embeddings.dtype).view(1, cond_dim).repeat(B, 1)
    embeddings = torch.cat([embeddings, c], dim=1)
    with torch.no_grad():
        pred_images = final_model(embeddings.to(device))
    # we have to denorm and clamp the pred_images to make comparable to images
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    pred_images = torch.clamp(pred_images * std + mean, 0, 1).cpu().to(device)
    # keep same checks at end of function, most important part
    assert pred_images.shape == (10, 3, 96, 128), f"Got {tuple(pred_images.shape)}"
    return torch.allclose(pred_images, images, atol=1e-5)

if __name__ == '__main__':
    print(test_load_my_models())
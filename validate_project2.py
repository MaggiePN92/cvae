"""
We will import this script for testing your models.
You have to write the function load_my_models(). It 
should take no arguments and return your final model.

The model should accept inputs matching your embedding
tensors, stored as embeddings.pth in the output.
The outputs should be of shape [10, 3, 96, 128].

Run this script with 'python validate_project2.py' to 
check that your implementation of load_my_models() works 
as intended.
"""

# Import modules for your own code
import torch
import torch.nn as nn

# Get your fine-tuned models here.
def load_my_models() -> nn.Module:
    """
    Loads your model.
    Use the imports to load the framework for your model.
    Then load the state dict such that your model is loaded
    with the correct weights.
    """
    ...

def test_load_my_models():
    final_model = load_my_models()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_model = final_model.to(device)

    final_model.eval()

    # Send an example through the models, to check that they loaded properly
    images = torch.load('output/images.pth')
    embeddings = torch.load('output/embeddings.pth')
    with torch.no_grad():
        pred_images = final_model(embeddings.to(device))

    return torch.allclose(pred_images, images, atol=1e-5)

if __name__ == '__main__':
    test_load_my_models()
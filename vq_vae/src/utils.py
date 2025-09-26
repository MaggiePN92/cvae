from pathlib import Path
import random
import numpy as np
import torch

def preprocess_weights(c, idxs):
    # Restructure and normalize chosen index's
    if c.dim() == 1:
        c = c.unsqueeze(1) # [B] -> [B, 1]
    c = c.view(c.size(0), -1)# [B, 1, 1, N] -> s[B, N]
    scores = c[:, idxs] # [B, 2]

    return scores

def get_idxmap(vq_model, x):
    # Return the embedding of x as idx from the codebook
    with torch.no_grad():
        # Encode the values from x to an embedding
        z_e = vq_model.encoder(x)
        _,_, enc_idx = vq_model.embedding(z_e)
        B = x.size(0)
        # Reshape to a grid: [B*H_l,W_l] -> [B, H_l, W_l]
        idx_map = enc_idx.view(B, z_e.shape[2], z_e.shape[3])
    return idx_map

def set_seed(seed=27):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_model(model, path, name, config : dict, extra : dict):
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok=True)
    torch.save({
        "model_name" : name,
        "state" : model.state_dict(),
        "config" : config if config else {},
        "extra" : extra if extra else {}
    }, path)
    
def load_model(model, path, map_location, strict = True):
    path = Path(path)
    payload = torch.load(path, map_location = map_location)

    state_dict = payload.get("state")
    if state_dict is None:
        raise KeyError(f"No 'state' key in stored payload for the model path {path}")
    model.load_state_dict(state_dict, strict = strict)

    return {
        "model_name" : payload.get("model_name"),
        "config" : payload.get("config", {}),
        "extra" : payload.get("extra", {})
    }

def get_loss_e(extra : dict):
    # Same signature as in ModelTracker
    epochs = extra["epochs"]
    losses = extra["losses"]
    durations = extra["durations"]
    
    return epochs, losses, durations

def unnormalize_tensor(t, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
    # Reverses the initial normalization
    mean = torch.as_tensor(mean, dtype=t.dtype, device=t.device).view(1, -1, 1, 1)
    std  = torch.as_tensor(std,  dtype=t.dtype, device=t.device).view(1, -1, 1, 1)
    return (t * std + mean)


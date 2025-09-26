import os
import torch
from torchvision.utils import save_image

def generate_img(model, z_dim, device, out_dir="generated_imgs", num_images=10, seed=None):
    model.eval()
    
    # random generator
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)

    # output directory
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        # scores for both moira and ferdinando
        target_condition = torch.tensor([1.0, 1.0], device=device).unsqueeze(0)
        
        generated_images = []
        latent_codes = []
        
        for i in range(num_images):
            # sample z
            z = torch.randn(1, z_dim, generator=g, device=device)
            
            c = target_condition
            
            # generate image
            z_cat = torch.cat([z, c], dim=1)
            x_hat = model.decoder(z_cat).squeeze(0).cpu()
            
            # denormalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            x_hat = x_hat * std + mean
            x_hat = torch.clamp(x_hat, 0, 1)
            
            filename = os.path.join(out_dir, f"moira_ferdinando_{i+1:02d}.png")
            save_image(x_hat, filename)
            print(f"generated {filename}")
            
            generated_images.append(x_hat)
            latent_codes.append(z.squeeze(0).cpu())
        
        # save as .pth files
        images_tensor = torch.stack(generated_images)
        embeddings_tensor = torch.stack(latent_codes)
        
        images_path = os.path.join(out_dir, "images.pth")
        torch.save(images_tensor, images_path)
        print(f"saved {images_path}")
        
        embeddings_data = {
            'latent_codes': embeddings_tensor,
            'target_condition': target_condition.cpu(),
            'condition_names': ['Moira', 'Ferdinando']
        }
        embeddings_path = os.path.join(out_dir, "embeddings.pth")
        torch.save(embeddings_data, embeddings_path)
        print(f"saved {embeddings_path}")
        
        print(f"generated {num_images} images")
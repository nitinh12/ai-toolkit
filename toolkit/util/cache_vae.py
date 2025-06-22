# toolkit/util/cache_vae.py
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def cache_vae_outputs(vae, dataset, output_dir, batch_size=4, device="cuda"):
    os.makedirs(output_dir, exist_ok=True)
    vae = vae.to(device)
    vae.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for batch in tqdm(dataloader, desc="Caching VAE outputs"):
        images = batch["image"].to(device)
        paths = batch["image_path"]

        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample()

        for latent, path in zip(latents, paths):
            filename = os.path.splitext(os.path.basename(path))[0] + ".pt"
            torch.save({"latents": latent.cpu()}, os.path.join(output_dir, filename))

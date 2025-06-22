# toolkit/util/cache_text.py
import os
import torch
from tqdm import tqdm

def cache_text_embeddings(text_encoder, tokenizer, prompts, output_dir, batch_size=16, device="cuda"):
    os.makedirs(output_dir, exist_ok=True)
    text_encoder = text_encoder.to(device)
    text_encoder.eval()

    for i in tqdm(range(0, len(prompts), batch_size), desc="Caching text embeddings"):
        batch_prompts = prompts[i:i+batch_size]
        tokens = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            outputs = text_encoder(**tokens)
            embeds = outputs.last_hidden_state

        for j, embed in enumerate(embeds):
            fname = str(abs(hash(batch_prompts[j]))) + ".pt"
            torch.save(embed.cpu(), os.path.join(output_dir, fname))

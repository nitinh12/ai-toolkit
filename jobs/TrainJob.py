import json
import os
from jobs import BaseJob
from toolkit.kohya_model_util import load_models_from_stable_diffusion_checkpoint
from collections import OrderedDict
from typing import List
from jobs.process import BaseExtractProcess, TrainFineTuneProcess
from datetime import datetime

# ✅ import our new caching utilities
from toolkit.util.cache_text import cache_text_embeddings
from toolkit.util.cache_vae import cache_vae_outputs
from toolkit.util.memory import reclaim_memory
from toolkit.data_transfer_object.data_loader import FlatPromptImageDataset

process_dict = {
    'vae': 'TrainVAEProcess',
    'slider': 'TrainSliderProcess',
    'slider_old': 'TrainSliderProcessOld',
    'lora_hack': 'TrainLoRAHack',
    'rescale_sd': 'TrainSDRescaleProcess',
    'esrgan': 'TrainESRGANProcess',
    'reference': 'TrainReferenceProcess',
}

class TrainJob(BaseJob):

    def __init__(self, config: OrderedDict):
        super().__init__(config)
        self.training_folder = self.get_conf('training_folder', required=True)
        self.is_v2 = self.get_conf('is_v2', False)
        self.device = self.get_conf('device', 'cpu')
        self.log_dir = self.get_conf('log_dir', None)

        # loads the processes from the config
        self.load_processes(process_dict)

    def run(self):
        super().run()
        print("")
        print(f"Running  {len(self.process)} process{'' if len(self.process) == 1 else 'es'}")

        # ✅ Add this block BEFORE any training starts
        if self.config.get("cache_preprocessing", True):
            try:
                process = self.process[0]  # Assume first process has the dataset, encoders, etc.
                if hasattr(process, "train_dataloader") and hasattr(process, "text_encoder") and hasattr(process, "vae"):
                    flat_dataset = FlatPromptImageDataset(process.train_dataloader.file_items)

                    # Cache text encoder embeddings
                    print("\U0001F4DD Caching text embeddings...")
                    cache_text_embeddings(
                        text_encoder=process.text_encoder,
                        tokenizer=process.tokenizer,
                        prompts=flat_dataset.get_all_prompts(),
                        output_dir=self.config["text_cache_dir"],
                        batch_size=16,
                        device=self.device,
                    )
                    process.text_encoder = None
                    reclaim_memory()

                    # Cache VAE latents
                    print("\U0001F5BC️ Caching VAE latents...")
                    cache_vae_outputs(
                        vae=process.vae,
                        dataset=flat_dataset,
                        output_dir=self.config["vae_cache_dir"],
                        batch_size=4,
                        device=self.device,
                    )
                    process.vae = None
                    reclaim_memory()

                    print("✅ Finished preprocessing cache. Continuing to training...")

                else:
                    print("⚠️ Skipping caching: Required attributes not found on training process.")

            except Exception as e:
                print(f"⚠️ Caching failed: {e}. Proceeding without caching.")

        # 👇 Original training loop
        for process in self.process:
            process.run()

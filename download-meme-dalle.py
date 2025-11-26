import os
from diffusers import StableDiffusionPipeline
import torch

def download_memesdalle(
    model_id="digiplay/incursiosMemeDiffusion_v1.6",
    save_dir="memesdalle_model"
):
    """
    Downloads the MemeDiffusion (Memes-DALL-E style) model locally.

    model_id: HuggingFace model name
    save_dir: folder where the model will be saved
    """

    os.makedirs(save_dir, exist_ok=True)

    print(f"[+] Downloading Meme model: {model_id}")
    print(f"[+] Saving to: {save_dir}")

    # Load and download the model
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    # Save the complete model to disk
    pipe.save_pretrained(save_dir)

    print("\n[✓] Download complete!")
    print("[✓] Model saved at:", os.path.abspath(save_dir))
    print("\nTo load the model later, use:\n")
    print(f'from diffusers import StableDiffusionPipeline\n'
          f'pipe = StableDiffusionPipeline.from_pretrained("{save_dir}")')

if __name__ == "__main__":
    download_memesdalle()


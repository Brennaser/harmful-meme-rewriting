import os
import argparse

import torch
import pandas as pd
from PIL import Image

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from huggingface_hub import login


SYSTEM_PROMPT = (
    "You are a content safety editor for internet memes.\n"
    "Your job is to rewrite short meme text so that it is safe and non offensive "
    "while keeping the original meaning, target, and joke structure as much as possible.\n\n"
    "Rules:\n"
    "1. Remove or soften slurs, insults, and explicit hate toward any group or person.\n"
    "2. Keep the same basic situation, characters, and point of view.\n"
    "3. Keep the text short, punchy, and meme like.\n"
    "4. Do not add new events or new facts. Small filler words are fine.\n"
    "5. If the input text is already safe and non offensive, return it unchanged.\n"
    "6. Reply with the rewritten meme text only."
)


def setup_hf_login(token: str | None = None):
    if token is None:
        token = os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise ValueError("Set HUGGINGFACE_TOKEN env var or pass token")
    login(token)


class GemmaBlipRewriter:
    def __init__(
        self,
        gemma_model_id: str = "google/gemma-2-2b-it",
        device: str | None = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print("Loading BLIP captioning model...")
        self.blip_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device)

        print("Loading Gemma model...")
        self.tokenizer = AutoTokenizer.from_pretrained(gemma_model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            gemma_model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        ).to(self.device)

        self.model_name = gemma_model_id

    def caption_image(self, image_path: str) -> str:
        image = Image.open(image_path).convert("RGB")
        inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.blip_model.generate(**inputs, max_new_tokens=40)
        caption = self.blip_processor.decode(out[0], skip_special_tokens=True).strip()
        return caption

    def rewrite(self, image_path: str, text: str) -> str:
        caption = self.caption_image(image_path)

        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Image description: {caption}\n"
            f"Original meme text: {text}\n\n"
            "Task:\nRewrite this text so that it is safe and non offensive, "
            "but still funny and as close as possible to the original meaning.\n"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
            )

        full = self.tokenizer.decode(out[0], skip_special_tokens=True)
        rewritten = full[len(prompt):].strip()
        if not rewritten:
            rewritten = full.strip()
        return rewritten


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="memes.csv")
    parser.add_argument("--output_csv", default="memes_gemma2_blip.csv")
    parser.add_argument("--gemma_model_id", default="google/gemma-2-2b-it")
    args = parser.parse_args()

    setup_hf_login()

    rewriter = GemmaBlipRewriter(gemma_model_id=args.gemma_model_id)
    df = pd.read_csv(args.input_csv)

    rewrites = []
    for idx, row in df.iterrows():
        image_path = row["image_path"]
        text = row["text"]
        print(f"[Gemma2 BLIP] Row {idx} image {image_path}")

        try:
            new_text = rewriter.rewrite(image_path, text)
        except Exception as e:
            print(f"Error on row {idx}: {e}")
            new_text = ""
        rewrites.append(new_text)

    df["gemma2_blip_rewrite"] = rewrites
    df.to_csv(args.output_csv, index=False)
    print(f"Saved Gemma plus BLIP outputs to {args.output_csv}")


if __name__ == "__main__":
    main()

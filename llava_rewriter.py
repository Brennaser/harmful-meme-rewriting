import os
import argparse
import json

import torch
import pandas as pd
from PIL import Image

from transformers import AutoProcessor
from transformers import LlavaForConditionalGeneration
from huggingface_hub import login
from typing import Union


SYSTEM_PROMPT = (
    "You are a content safety editor for internet memes.\n"
    "Your job is to rewrite short meme text and generate new images so that the new meme is safe and non offensive "
    "while keeping the original meaning, target, and joke structure as much as possible.\n\n"
    "Rules:\n"
    "1. Remove or soften slurs, insults, and explicit hate toward any group or person.\n"
    "2. Keep the same basic situation, characters, and point of view.\n"
    "3. Keep the text short, punchy, and meme like.\n"
    "4. Reply with the rewritten meme text only."
)


def setup_hf_login(token: Union[str, None] = None):
    if token is None:
        token = os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise ValueError("Set HUGGINGFACE_TOKEN env var or pass token")
    login(token)


class LlavaRewriter:
    def __init__(
        self,
        llava_model_id: str = "llava-hf/llava-1.5-7b-hf",
        device: Union[str, None] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print("Loading LLaVA model...")
        self.processor = AutoProcessor.from_pretrained(llava_model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            llava_model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        self.model_name = llava_model_id

    def rewrite(self, image_path: str, text: str) -> str:
        image = Image.open(image_path).convert("RGB")

        user_instruction = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Original meme text: {text}\n\n"
        )

        prompt = f"USER: <image>\n{user_instruction}\nASSISTANT:"

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
            )

        generated_ids = out[0]
        text_out = self.processor.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        if "ASSISTANT:" in text_out:
            text_out = text_out.split("ASSISTANT:", 1)[-1].strip()

        return text_out.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="memes.csv")
    parser.add_argument("--output_csv", default="memes_llava.csv")
    parser.add_argument("--llava_model_id", default="llava-hf/llava-1.5-7b-hf")
    args = parser.parse_args()

    setup_hf_login()

    rewriter = LlavaRewriter(llava_model_id=args.llava_model_id)
    df = pd.read_json(args.input_csv, lines=True)

    rewrites = []
    for idx, row in df.iterrows():
        image_path = row["img"]
        text = row["text"]
        print(f"[LLaVA] Row {idx} image {image_path}")

        try:
            new_text = rewriter.rewrite(image_path, text)
        except Exception as e:
            print(f"Error on row {idx}: {e}")
            new_text = ""
        rewrites.append(new_text)

    df["llava_rewrite"] = rewrites
    df.to_csv(args.output_csv, index=False)
    print(f"Saved LLaVA outputs to {args.output_csv}")


if __name__ == "__main__":
    main()

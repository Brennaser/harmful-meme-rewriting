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


BASIC_PROMPT = (
    "You are a content safety editor for internet memes.\n"
    "Your task is to rewrite meme text and suggest a new image description so that the meme becomes safe and non offensive.\n\n"
    "Rules:\n"
    "1. Remove or soften any slurs, insults, or explicit hate toward groups or individuals.\n"
    "2. Keep the rewritten text short, punchy, and meme like.\n"
    "3. Avoid adding new characters, new political content, or new offensive ideas.\n"
    "4. Keep the core joke or message understandable after editing.\n"
    "5. Output two fields only:\n"
    "   Safe_Text: the rewritten meme text\n"
    "   Safe_Image: a short description of a safe image replacement.\n"
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

        # BASIC + few shot examples
        user_instruction = (
            f"{BASIC_PROMPT}\n\n"
            "Here are some examples of how to rewrite unsafe memes into safe ones:\n\n"
            "Example 1:\n"
            "Original: \"when you forget you're retarded\"\n"
            "Safe_Text: \"when you forget why you walked into the room\"\n"
            "Safe_Image: \"A person standing in a room looking confused about what they came in for\"\n"
            "Explanation: The rewrite removes the slur and turns it into a relatable forgetful moment for anyone.\n\n"
            "Example 2:\n"
            "Original: \"i'm not a racist my shadow is black\"\n"
            "Safe_Text: \"I try to appreciate everyone and their differences\"\n"
            "Safe_Image: \"A diverse group of friends smiling together\"\n"
            "Explanation: The rewrite removes the racist logic and replaces it with an inclusive positive message.\n\n"
            "Now rewrite the following meme in the same style:\n\n"
            f"Original: \"{text}\"\n\n"
            "Return your answer in exactly this format:\n"
            "Safe_Text: ...\n"
            "Safe_Image: ...\n"
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
                max_new_tokens=80,
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
    parser.add_argument("--input_jsonl", default="train190_subset.jsonl")
    parser.add_argument("--output_csv", default="memes_llava.csv")
    parser.add_argument("--llava_model_id", default="llava-hf/llava-1.5-7b-hf")
    args = parser.parse_args()

    setup_hf_login()

    rewriter = LlavaRewriter(llava_model_id=args.llava_model_id)
    df = pd.read_json(args.input_jsonl, lines=True)

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

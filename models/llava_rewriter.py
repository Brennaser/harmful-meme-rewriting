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
    """
    Log in to Hugging Face Hub.
    """
    if token is None:
        token = os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        print("Warning: HUGGINGFACE_TOKEN not set. Proceeding without explicit login.")
    else:
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

        print(f"Loading LLaVA model {llava_model_id} on device {self.device}...")
        self.processor = AutoProcessor.from_pretrained(llava_model_id)
        device_map_arg = "auto" if self.device == "cuda" else None
        torch_dtype_arg = torch.float16 if self.device == "cuda" else torch.float32

        self.model = LlavaForConditionalGeneration.from_pretrained(
            llava_model_id,
            torch_dtype=torch_dtype_arg,
            device_map=device_map_arg,
        )
        self.model_name = llava_model_id

    def rewrite(self, image_path: str, text: str) -> str:
        """
        Rewrites a single meme based on the image and text, using a zero-shot prompt.
        """
        image = Image.open(image_path).convert("RGB")

        # Zero-shot prompt construction
        user_instruction = (
            f"{BASIC_PROMPT}\n\n"
            f"Now rewrite the following meme based on the image and text:\n\n"
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
    parser = argparse.ArgumentParser(description="Run LLaVA for zero-shot meme rewriting from a JSONL file.")
    
    # Arguments for batch processing
    parser.add_argument("--input_jsonl", 
                        required=True, 
                        help="Path to the JSONL file containing meme data (id, img, text).")
    parser.add_argument("--image_dir", 
                        required=True, 
                        help="Base directory where all image files are located (e.g., '/home/kiwi-pandas/Documents/harmful-meme-rewriting/img').")
    parser.add_argument("--output_csv", 
                        default="llava_rewrites.csv", 
                        help="Path to save the output CSV file with rewrites.")
    
    # Model argument
    parser.add_argument("--llava_model_id", 
                        default="llava-hf/llava-1.5-7b-hf",
                        help="Hugging Face ID of the LLaVA model to use")
    
    args = parser.parse_args()

    setup_hf_login()

    rewriter = LlavaRewriter(llava_model_id=args.llava_model_id)
    
    try:
        # Read the JSONL file into a pandas DataFrame
        df = pd.read_json(args.input_jsonl, lines=True)
    except FileNotFoundError:
        print(f"Error: JSONL file not found at {args.input_jsonl}")
        return

    rewrites = []
    
    print(f"Processing {len(df)} entries from {args.input_jsonl}...")

    for idx, row in df.iterrows():
        # The 'img' column might only contain 'img/15803.png', so we join it with the base directory
        # We also need to strip the 'img/' prefix if it exists to correctly join with the base directory
        relative_path = row["img"].replace("img/", "")
        image_path = os.path.join(args.image_dir, relative_path)
        
        text = row["text"]
        
        print(f"[{idx+1}/{len(df)}] ID: {row['id']} | Image: {image_path} | Text: \"{text[:40]}...\"")

        try:
            # Pass the full image path and text to the rewrite method
            new_text = rewriter.rewrite(image_path, text)
        except FileNotFoundError:
            print(f"   --> Error: Image file not found at {image_path}. Skipping.")
            new_text = "ERROR: Image file not found"
        except Exception as e:
            print(f"   --> Error on row {idx}: {e}")
            new_text = f"ERROR: {str(e)}"
            
        rewrites.append(new_text)

    df["llava_rewrite"] = rewrites
    df.to_csv(args.output_csv, index=False)
    print(f"\n✅ Processing complete. Saved LLaVA outputs to {args.output_csv}")


if __name__ == "__main__":
    main()
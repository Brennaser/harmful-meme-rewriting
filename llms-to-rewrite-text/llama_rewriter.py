import os
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from typing import Union

# ------------------------------------------------------
# PROMPTS - Few-shot approach from Gemini
# ------------------------------------------------------

BASIC_PROMPT = (
    "You are a content safety editor for internet memes.\n"
    "Your task is to rewrite meme text so that the meme becomes safe and non-offensive.\n\n"
    "Rules:\n"
    "1. Remove or soften any slurs, insults, or explicit hate toward groups or individuals.\n"
    "2. Keep the rewritten text short, punchy, and meme-like.\n"
    "3. Avoid adding new characters, new political content, or new offensive ideas.\n"
    "4. Keep the core joke or message understandable after editing.\n"
    "5. Output only the rewritten safe text, nothing else."
)

FEW_SHOT_PROMPT = (
    f"{BASIC_PROMPT}\n\n"
    "Here are some examples of how to rewrite unsafe memes into safe ones:\n\n"
    "Example 1:\n"
    "Original: \"when you forget you're retarded\"\n"
    "Safe_Text: \"when you forget why you walked into the room\"\n\n"
    "Example 2:\n"
    "Original: \"i'm not a racist my shadow is black\"\n"
    "Safe_Text: \"I try to appreciate everyone and their differences\"\n\n"
    "Example 3:\n"
    "Original: \"disabled people be like\"\n"
    "Safe_Text: \"when you try something new for the first time\"\n\n"
    "Now rewrite the following meme text in the same style:\n"
)


# ------------------------------------------------------
# SETUP HUGGINGFACE LOGIN
# ------------------------------------------------------

def setup_hf_login(token: Union[str, None] = None):
    if token is None:
        token = os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise ValueError("Set HUGGINGFACE_TOKEN env var or pass token")
    login(token)


# ------------------------------------------------------
# LLAMA REWRITER CLASS
# ------------------------------------------------------

class LlamaRewriter:
    def __init__(
            self,
            model_id: str = "meta-llama/Llama-2-7b-chat-hf",
            device: Union[str, None] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"Loading Llama model: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        self.model_name = model_id

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def rewrite(self, text: str, use_few_shot: bool = True) -> str:
        """Rewrite meme text to be safe and non-offensive."""

        # Choose prompt based on few-shot setting
        if use_few_shot:
            prompt = f"{FEW_SHOT_PROMPT}Original: \"{text}\"\nSafe_Text: "
        else:
            prompt = f"{BASIC_PROMPT}\n\nOriginal: \"{text}\"\nSafe_Text: "

        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        # Generate output
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the rewritten text (after "Safe_Text: ")
        if "Safe_Text: " in generated_text:
            rewrite = generated_text.split("Safe_Text: ", 1)[-1].strip()
        else:
            rewrite = generated_text.strip()

        # Clean up any trailing special tokens or artifacts
        rewrite = rewrite.split("\n")[0].strip()

        return rewrite


# ------------------------------------------------------
# MAIN SCRIPT
# ------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", default="img/train190_subset.jsonl")
    parser.add_argument("--output_csv", default="results/memes_llama_chat.csv")
    parser.add_argument(
        "--model_id",
        default="meta-llama/Llama-2-7b-chat-hf",
        help="Hugging Face model ID for Llama"
    )
    parser.add_argument(
        "--use_few_shot",
        action="store_true",
        default=True,
        help="Use few-shot prompting (default: True)"
    )
    args = parser.parse_args()

    setup_hf_login()

    rewriter = LlamaRewriter(model_id=args.model_id)
    df = pd.read_json(args.input_jsonl, lines=True)

    rewrites = []
    for idx, row in df.iterrows():
        text = row["text"]
        print(f"[Llama] Row {idx}: {text[:50]}...")

        try:
            new_text = rewriter.rewrite(text, use_few_shot=args.use_few_shot)
        except Exception as e:
            print(f"Error on row {idx}: {e}")
            new_text = ""

        rewrites.append(new_text)

    df["llama_rewrite"] = rewrites
    df.to_csv(args.output_csv, index=False)
    print(f"Saved Llama outputs to {args.output_csv}")


if __name__ == "__main__":
    main()
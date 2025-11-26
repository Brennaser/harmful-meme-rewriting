import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm

import os
from huggingface_hub import login
print("Token in env:", os.environ.get("HUGGINGFACE_TOKEN") is not None)
login(token=os.environ["HUGGINGFACE_TOKEN"])




# ===========================
# PROMPTS
# ===========================

BASIC_PROMPT = (
    "You are a content safety editor for internet memes.\n"
    "Your task is to rewrite meme text and propose a new image description so that the meme becomes safe and non offensive.\n\n"
    "Rules:\n"
    "1. Remove or soften any slurs, insults, or explicit hate toward groups or individuals.\n"
    "2. Keep the rewritten text short, punchy, and meme like.\n"
    "3. Avoid adding new characters, new political content, or new offensive ideas.\n"
    "4. Keep the core joke or message understandable after editing.\n"
    "5. Output two fields only:\n"
    "   Safe_Text: the rewritten text\n"
    "   Safe_Image: a short description of a safe replacement image."
)

FEW_SHOT_PROMPT = (
    f"{BASIC_PROMPT}\n\n"
    "Here are some examples of how to rewrite unsafe memes into safe ones:\n\n"
    "Example 1:\n"
    "Original: \"when you forget you're retarded\"\n"
    "Safe_Text: \"when you forget why you walked into the room\"\n"
    "Safe_Image: \"A person standing in a room looking confused about what they came in for\"\n\n"
    "Example 2:\n"
    "Original: \"i'm not a racist my shadow is black\"\n"
    "Safe_Text: \"I try to appreciate everyone and their differences\"\n"
    "Safe_Image: \"A diverse group of friends smiling together\"\n\n"
    "Now rewrite the following meme in the same style:\n"
)


# ===========================
# Helper functions
# ===========================

def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def save_jsonl(items, path):
    with open(path, "w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def load_model(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    return tokenizer, model, device


def run_model(tokenizer, model, device, prompt, input_text, max_new_tokens):
    full = (
        prompt
        + "\nOriginal:\n"
        + input_text
        + "\n\nAnswer:\n"
    )

    encoded = tokenizer(full, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    if decoded.startswith(full):
        decoded = decoded[len(full):]

    return decoded.strip()


# ===========================
# Main script
# ===========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", default="img/train190_subset.jsonl")
    parser.add_argument("--output_csv", default="memes_gemma_basic.csv")
    parser.add_argument("--model_name", default="google/gemma-2-2b")
    parser.add_argument("--max_new_tokens", default=150, type=int)
    parser.add_argument("--mode", choices=["basic", "fewshot"], default="basic")
    args = parser.parse_args()

    data = load_jsonl(args.input_jsonl)
    print(f"Loaded {len(data)} memes")

    tokenizer, model, device = load_model(args.model_name)

    prompt = BASIC_PROMPT if args.mode == "basic" else FEW_SHOT_PROMPT

    results = []
    for item in tqdm(data, desc=f"Running Gemma2 ({args.mode})"):
        text = item.get("text", "")

        try:
            generation = run_model(
                tokenizer, model, device, prompt, text, args.max_new_tokens
            )
        except Exception as e:
            generation = f"ERROR: {e}"

        new_item = dict(item)
        new_item[f"gemma2_{args.mode}"] = generation
        results.append(new_item)

    save_csv(results, args.output_jsonl)
    print(f"Saved output to {args.output_csv}")


if __name__ == "__main__":
    main()

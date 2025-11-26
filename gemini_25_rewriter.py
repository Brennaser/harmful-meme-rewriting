import os
import argparse
import pandas as pd
import google.generativeai as genai


# ------------------------------------------------------
# PROMPTS
# ------------------------------------------------------

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


# ------------------------------------------------------
# GEMINI CLIENT
# ------------------------------------------------------

def get_gemini_client(model_name: str, api_key: str | None = None):
    if api_key is None:
        api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("You must set GEMINI_API_KEY")

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


# ------------------------------------------------------
# MAIN REWRITE FUNCTION
# ------------------------------------------------------

def rewrite_with_gemini(model, image_path: str, text: str, prompt_mode: str):
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # Pick which prompt to use
    if prompt_mode == "basic":
        final_prompt = (
            f"{BASIC_PROMPT}\n\n"
            f"Original: \"{text}\"\n\n"
            "Return your answer exactly in this format:\n"
            "Safe_Text: ...\n"
            "Safe_Image: ...\n"
        )
    else:
        # few-shot version
        final_prompt = (
            f"{FEW_SHOT_PROMPT}"
            f"Original: \"{text}\"\n\n"
            "Return your answer exactly in this format:\n"
            "Safe_Text: ...\n"
            "Safe_Image: ...\n"
        )

    response = model.generate_content(
        [
            {"mime_type": "image/png", "data": image_bytes},
            final_prompt,
        ]
    )

    return response.text.strip()


# ------------------------------------------------------
# MAIN SCRIPT
# ------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", default="img\\train190_subset.jsonl")
    parser.add_argument("--output_csv", default="memes_gemini_v2.csv")
    parser.add_argument("--model_name", default="gemini-2.5-flash")
    parser.add_argument("--image_root", default=".")
    parser.add_argument(
        "--prompt_mode",
        default="fewshot",
        choices=["basic", "fewshot"],
        help="Choose which prompt style to use"
    )
    args = parser.parse_args()

    model = get_gemini_client(args.model_name)

    df = pd.read_json(args.input_jsonl, lines=True)

    outputs = []
    for idx, row in df.iterrows():
        image_path = os.path.join(args.image_root, row["img"])
        text = row["text"]

        print(f"[Gemini] Row {idx} using {args.prompt_mode} prompt...")

        try:
            rewrite = rewrite_with_gemini(
                model=model,
                image_path=image_path,
                text=text,
                prompt_mode=args.prompt_mode,
            )
        except Exception as e:
            print(f"Error on row {idx}: {e}")
            rewrite = ""

        outputs.append(rewrite)

    df["gemini_rewrite"] = outputs
    df.to_csv(args.output_csv, index=False)

    print(f"Saved to {args.output_csv}")


if __name__ == "__main__":
    main()

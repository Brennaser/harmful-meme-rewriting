import os
import argparse
import pandas as pd

import google.generativeai as genai  # Google Gen AI SDK

BASIC_PROMPT = (
    "You are a content safety editor for internet memes.\n"
    "Your task is to rewrite meme text and propose a new image description so that the meme becomes safe and non offensive.\n\n"
    "Rules:\n"
    "1. Remove or soften any slurs, insults, or explicit hate toward groups or individuals.\n"
    "2. Keep the rewritten text short, punchy, and meme like.\n"
    "3. Avoid adding new characters, new political content, or new offensive ideas.\n"
    "4. Keep the core joke or message understandable after editing.\n"
    "5. Output two fields only:\n"
    " Safe_Text: the rewritten meme text\n"
    " Safe_Image: a short description of a safe image replacement."
)

def get_gemini_client(api_key: str | None = None):
    # The client will read GEMINI_API_KEY from the environment if not passed
    if api_key is None:
        api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY env var or pass api_key")
    # client = genai.Client(api_key=api_key)
    genai.configure(api_key=api_key)
    gemini = genai.GenerativeModel('gemini-2.5-flash-lite')

    return gemini


def rewrite_with_gemini(
    model,
    image_path: str,
    text: str,
    few_shot_example: bool,
) -> str:
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    if few_shot_example:
        pass
    else:
            user_prompt = (
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

    response = model.generate_content(
        contents=[
            {
                "role": "user",
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "image/png",  # change to image/jpeg if needed
                            "data": image_bytes,
                        }
                    },
                    {"text": user_prompt},
                ],
            }
        ],
    )

    return response.text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", default="train190_subset.jsonl")
    parser.add_argument("--output_csv", default="memes_gemini-v2.csv")
    parser.add_argument("--model_name", default="gemini-2.5-flash")
    args = parser.parse_args()

    client = get_gemini_client()
    df = pd.read_json(args.input_jsonl, lines=True)

    rewrites = []
    for idx, row in df.iterrows():
        image_path = row["img"]
        text = row["text"]
        print(f"[Gemini] Row {idx} image {image_path}")

        try:
            new_text = rewrite_with_gemini(client, args.model_name, image_path, text)
        except Exception as e:
            print(f"Error on row {idx}: {e}")
            new_text = ""
        rewrites.append(new_text)

    df["gemini_25_rewrite"] = rewrites
    df.to_csv(args.output_csv, index=False)
    print(f"Saved Gemini outputs to {args.output_csv}")


if __name__ == "__main__":
    main()

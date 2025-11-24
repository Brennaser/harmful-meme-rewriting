import os
import argparse
import pandas as pd

import google.generativeai as genai  # Google Gen AI SDK


BASIC_PROMPT = (
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
    model_name: str,
    image_path: str,
    text: str,
    chain_of_thought: bool,
) -> str:
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    if chain_of_thought:
        pass
    else :
        user_prompt = (
            f"{BASIC_PROMPT}\n\n"
            f"Original meme text:\n{text}\n\n"
            "Task:\nRewrite this text so that it is safe and non offensive, "
            "but still funny and as close as possible to the original meaning."
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
    parser.add_argument("--input_csv", default="memes.csv")
    parser.add_argument("--output_csv", default="memes_gemini.csv")
    parser.add_argument("--model_name", default="gemini-2.5-flash")
    args = parser.parse_args()

    client = get_gemini_client("AIzaSyCcdpjNAeNeN3a2wPSLtdcbra_3rjU37oE")
    df = pd.read_json(args.input_csv, lines=True)

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

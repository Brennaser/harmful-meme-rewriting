import os
import argparse
import pandas as pd
from anthropic import Anthropic
import time

# ------------------------------------------------------
# PROMPTS
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
# CLAUDE REWRITER CLASS
# ------------------------------------------------------

class ClaudeRewriter:
    def __init__(self, model: str = "claude-3-5-sonnet-20241022", api_key: str = None):
        """
        Initialize Claude rewriter.

        Args:
            model: Anthropic Claude model to use
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var
        """
        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")

        if not api_key:
            raise ValueError("Set ANTHROPIC_API_KEY env var or pass api_key")

        self.client = Anthropic(api_key=api_key)
        self.model = model
        print(f"Using Claude model: {model}")

    def rewrite(self, text: str, use_few_shot: bool = True) -> str:
        """Rewrite meme text to be safe and non-offensive."""

        # Choose prompt based on few-shot setting
        if use_few_shot:
            prompt = f"{FEW_SHOT_PROMPT}Original: \"{text}\"\nSafe_Text: "
        else:
            prompt = f"{BASIC_PROMPT}\n\nOriginal: \"{text}\"\nSafe_Text: "

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=100,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            rewrite = response.content[0].text.strip()
            return rewrite

        except Exception as e:
            print(f"Error generating rewrite: {e}")
            return ""


# ------------------------------------------------------
# MAIN SCRIPT
# ------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", default="img/train190_subset.jsonl")
    parser.add_argument("--output_csv", default="results/memes_claude.csv")
    parser.add_argument(
        "--model",
        default="claude-opus-4-1-20250805",
        help="Anthropic Claude model to use"
    )
    parser.add_argument(
        "--use_few_shot",
        action="store_true",
        default=True,
        help="Use few-shot prompting (default: True)"
    )
    args = parser.parse_args()

    rewriter = ClaudeRewriter(model=args.model)
    df = pd.read_json(args.input_jsonl, lines=True)

    rewrites = []
    for idx, row in df.iterrows():
        text = row["text"]
        print(f"[Claude] Row {idx}: {text[:50]}...")

        try:
            new_text = rewriter.rewrite(text, use_few_shot=args.use_few_shot)
        except Exception as e:
            print(f"Error on row {idx}: {e}")
            new_text = ""

        rewrites.append(new_text)
        time.sleep(1)

    df["claude_rewrite"] = rewrites
    df.to_csv(args.output_csv, index=False)
    print(f"Saved Claude outputs to {args.output_csv}")


if __name__ == "__main__":
    main()
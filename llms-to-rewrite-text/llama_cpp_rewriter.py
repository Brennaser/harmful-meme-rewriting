import os
import argparse
import pandas as pd
from llama_cpp import Llama
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
# LLAMA CPP REWRITER CLASS
# ------------------------------------------------------

class LlamaCppRewriter:
    def __init__(
            self,
            model_path: str = None,
            n_gpu_layers: int = 1,
    ):
        """
        Initialize Llama.cpp rewriter.

        Args:
            model_path: Path to GGUF model file. If None, will download Llama-2-7b-chat
            n_gpu_layers: Number of layers to offload to GPU (0 = CPU only, higher = more GPU)
        """

        if model_path is None:
            # For M1 Mac, use a GGUF quantized model
            model_path = "TheBloke/Llama-2-7B-Chat-GGUF"

        print(f"Loading Llama model from: {model_path}")

        try:
            self.llm = Llama(
                model_path=model_path,
                n_ctx=512,
                n_threads=4,  # Adjust based on your Mac cores
                n_gpu_layers=n_gpu_layers,  # Use GPU if available
                verbose=True,
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Make sure you have llama-cpp-python installed: pip install llama-cpp-python")
            raise

    def rewrite(self, text: str, use_few_shot: bool = True) -> str:
        """Rewrite meme text to be safe and non-offensive."""

        # Choose prompt based on few-shot setting
        if use_few_shot:
            prompt = f"{FEW_SHOT_PROMPT}Original: \"{text}\"\nSafe_Text: "
        else:
            prompt = f"{BASIC_PROMPT}\n\nOriginal: \"{text}\"\nSafe_Text: "

        try:
            # Generate output using llama.cpp
            output = self.llm(
                prompt,
                max_tokens=30,
                temperature=0.7,
                top_p=0.9,
                stop=["Original:", "\n"],
            )

            generated_text = output["choices"][0]["text"]

            # Extract only the rewritten text
            rewrite = generated_text.strip()

            # Clean up any trailing artifacts
            rewrite = rewrite.split("\n")[0].strip()

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
    parser.add_argument("--output_csv", default="results/memes_llama_cpp.csv")
    parser.add_argument(
        "--model_path",
        default="/Users/shambhaviverma/Downloads/llama-2-7b-chat.Q4_K_M.gguf",
        help="Path or HF model ID for GGUF model"
    )
    parser.add_argument(
        "--use_few_shot",
        action="store_true",
        default=True,
        help="Use few-shot prompting (default: True)"
    )
    parser.add_argument(
        "--n_gpu_layers",
        type=int,
        default=1,
        help="Number of layers to offload to GPU (0=CPU only, higher=more GPU)"
    )
    args = parser.parse_args()

    rewriter = LlamaCppRewriter(
        model_path=args.model_path,
        n_gpu_layers=args.n_gpu_layers,
    )

    df = pd.read_json(args.input_jsonl, lines=True)

    rewrites = []
    for idx, row in df.iterrows():
        text = row["text"]
        print(f"[Llama-CPP] Row {idx}: {text[:50]}...")

        try:
            new_text = rewriter.rewrite(text, use_few_shot=args.use_few_shot)
        except Exception as e:
            print(f"Error on row {idx}: {e}")
            new_text = ""

        rewrites.append(new_text)

    df["llama_cpp_rewrite"] = rewrites
    df.to_csv(args.output_csv, index=False)
    print(f"Saved Llama-CPP outputs to {args.output_csv}")


if __name__ == "__main__":
    main()
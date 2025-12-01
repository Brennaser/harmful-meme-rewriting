import os
import pandas as pd
import torch

from detoxify import Detoxify
from sentence_transformers import SentenceTransformer, util

# -----------------------------
# Path Setup (Linux-safe)
# -----------------------------
BASE = os.path.dirname(os.path.abspath(__file__))     # eval/
ROOT = os.path.abspath(os.path.join(BASE, ".."))      # repo root
RESULTS_DIR = os.path.join(ROOT, "results")
EVAL_DIR = os.path.join(ROOT, "eval")

os.makedirs(EVAL_DIR, exist_ok=True)

INPUT_FILES = {
    "gemini":         os.path.join(RESULTS_DIR, "memes_gemini.csv"),
    "gemini_fewshot": os.path.join(RESULTS_DIR, "memes_gemini_fewshot.csv"),
    "gemini_basic":   os.path.join(RESULTS_DIR, "memes_gemini_basic.csv"),
    "gemma_basic":    os.path.join(RESULTS_DIR, "memes_gemma_basic.csv"),
    "llava":          os.path.join(RESULTS_DIR, "memes_llava.csv"),
    "gpt":            os.path.join(RESULTS_DIR, "memes_chatgpt.csv"),
    "claude":         os.path.join(RESULTS_DIR, "memes_claude.csv"),
}

ID_COL = "id"
IMG_COL = "img"
LABEL_COL = "label"
ORIG_COL = "text"

NON_REWRITE_BASE = {ID_COL, IMG_COL, LABEL_COL, ORIG_COL}

# -----------------------------
# Model setup
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

sim_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)
tox_model = Detoxify("original", device=device)

# -----------------------------
# Helper Functions
# -----------------------------
def extract_safe_text(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    s = raw.strip()
    if "Safe_Text:" not in s:
        return s
    after = s.split("Safe_Text:", 1)[1]
    if "Safe_Image:" in after:
        after = after.split("Safe_Image:", 1)[0]
    return after.strip(" \n\t:-")


def clean_rewrite_column(df, rewrite_col):
    return df[rewrite_col].fillna("").astype(str).apply(extract_safe_text)


def find_rewrite_column(df, path):
    rewrite_candidates = [c for c in df.columns if c not in NON_REWRITE_BASE]
    if len(rewrite_candidates) == 0:
        raise ValueError(f"No rewrite column found in {path}")
    if len(rewrite_candidates) > 1:
        raise ValueError(f"Multiple rewrite columns detected in {path}: {rewrite_candidates}")
    return rewrite_candidates[0]


def add_toxicity(df, orig, new):
    print("  - Computing toxicity...")
    df["toxicity_before"] = tox_model.predict(df[orig].astype(str).tolist())["toxicity"]
    df["toxicity_after"]  = tox_model.predict(df[new].astype(str).tolist())["toxicity"]
    df["toxicity_reduction"] = df["toxicity_before"] - df["toxicity_after"]
    return df


def add_similarity(df, orig, new):
    print("  - Computing semantic similarity...")
    emb_orig = sim_model.encode(df[orig].astype(str).tolist(), convert_to_tensor=True, show_progress_bar=True)
    emb_new  = sim_model.encode(df[new].astype(str).tolist(), convert_to_tensor=True, show_progress_bar=True)
    df["semantic_similarity"] = util.cos_sim(emb_orig, emb_new).diagonal().cpu().numpy()
    return df


def summarize(df, model_name):
    return {
        "model": model_name,
        "mean_toxicity_before": df["toxicity_before"].mean(),
        "mean_toxicity_after": df["toxicity_after"].mean(),
        "mean_toxicity_reduction": df["toxicity_reduction"].mean(),
        "mean_semantic_similarity": df["semantic_similarity"].mean(),
        "std_semantic_similarity": df["semantic_similarity"].std(),
        "n_examples": len(df),
    }

# -----------------------------
# Main
# -----------------------------
def main():
    summaries = []

    for model_name, path in INPUT_FILES.items():
        if not os.path.exists(path):
            print(f"Skipping {model_name} (file not found): {path}")
            continue

        print(f"\n=== Evaluating {model_name} ===")
        df = pd.read_csv(path)

        if ORIG_COL not in df.columns:
            raise ValueError(f"Original text column '{ORIG_COL}' not found in {path}")

        rewrite_col = find_rewrite_column(df, path)
        print(f"  - Detected rewrite column: {rewrite_col}")

        df["rewrite_clean"] = clean_rewrite_column(df, rewrite_col)

        df = add_toxicity(df, ORIG_COL, "rewrite_clean")
        df = add_similarity(df, ORIG_COL, "rewrite_clean")

        out_file = os.path.join(EVAL_DIR, f"{model_name}_evaluated.csv")
        df.to_csv(out_file, index=False)
        print(f"  -> Saved: {out_file}")

        summaries.append(summarize(df, model_name))

    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_path = os.path.join(EVAL_DIR, "model_metrics_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        print("\n=== Model Comparison Summary ===")
        print(summary_df)
        print(f"\nSummary saved to {summary_path}")

        # Additional comparison: Gemini vs Few-shot
        if {"gemini", "gemini_fewshot"}.issubset(summary_df["model"].unique()):
            print("\n=== Gemini vs Few-Shot Comparison ===")
            reg = summary_df[summary_df["model"] == "gemini"].iloc[0]
            fs  = summary_df[summary_df["model"] == "gemini_fewshot"].iloc[0]

            compare = pd.DataFrame([
                {"metric": "toxicity_before", "gemini": reg["mean_toxicity_before"], 
                 "few_shot": fs["mean_toxicity_before"]},
                {"metric": "toxicity_after", "gemini": reg["mean_toxicity_after"], 
                 "few_shot": fs["mean_toxicity_after"]},
                {"metric": "toxicity_reduction", "gemini": reg["mean_toxicity_reduction"], 
                 "few_shot": fs["mean_toxicity_reduction"]},
                {"metric": "semantic_similarity", "gemini": reg["mean_semantic_similarity"], 
                 "few_shot": fs["mean_semantic_similarity"]},
            ])

            print(compare.to_string(index=False))
            compare.to_csv(os.path.join(EVAL_DIR, "gemini_vs_fewshot_summary.csv"), index=False)
            print("\nGemini vs Few-shot comparison saved in eval/")

if __name__ == "__main__":
    main()

import os
import pandas as pd
import torch

from detoxify import Detoxify
from sentence_transformers import SentenceTransformer, util

# -----------------------------
# Config
# -----------------------------
# Map model name -> CSV path
INPUT_FILES = {
    "gemini": "results/memes_gemini.csv",
    "gemini_fewshot": "results/memes_gemini_fewshot.csv",
    "llama": "results/memes_llama.csv",   # if you want it
    "gemma": "results/memes_gemma_basic.csv",
    "llava": "results/memes_llava.csv",
    "gpt": "results/memes_gpt.csv",
    "claude": "results/memes_claude.csv",
}

ID_COL = "id"
IMG_COL = "img"
LABEL_COL = "label"
ORIG_COL = "text"   # original meme text column name

# Columns that are NOT rewrite candidates
NON_REWRITE_BASE = {ID_COL, IMG_COL, LABEL_COL, ORIG_COL}

# -----------------------------
# Model setup
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

sim_model = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2",
    device=device
)
tox_model = Detoxify("original", device=device)


# -----------------------------
# Helpers
# -----------------------------
def extract_safe_text(raw: str) -> str:
    """
    Extract just the Safe_Text portion from a combined field like:
      'Safe_Text: ... Safe_Image: ...'

    If 'Safe_Text:' is not present, returns the original string stripped.
    This works for:
      - Gemini runs where you have Safe_Text + Safe_Image
      - Plain rewrites from GPT, Claude, LLaVA, etc (no tags)
    """
    if not isinstance(raw, str):
        return ""

    s = raw.strip()
    if "Safe_Text:" not in s:
        return s

    # Take portion after 'Safe_Text:'
    after = s.split("Safe_Text:", 1)[1]

    # If there's a Safe_Image tag, cut before it
    if "Safe_Image:" in after:
        after = after.split("Safe_Image:", 1)[0]

    return after.strip(" \n\t:-")


def clean_rewrite_column(df: pd.DataFrame, rewrite_col: str) -> pd.Series:
    """
    Returns a cleaned version of the rewrite column where, if present,
    we strip out everything except the Safe_Text portion.
    """
    return df[rewrite_col].fillna("").astype(str).apply(extract_safe_text)


def find_rewrite_column(df: pd.DataFrame, path: str) -> str:
    """
    Infer the rewrite column as the one that is not id/img/label/text.
    Raises if it finds zero or more than one candidate.
    """
    cols = set(df.columns)
    rewrite_candidates = [c for c in cols if c not in NON_REWRITE_BASE]

    if len(rewrite_candidates) == 0:
        raise ValueError(
            f"No rewrite column found in {path}. "
            f"Columns are: {list(df.columns)}. "
            f"Expected something besides {NON_REWRITE_BASE}."
        )
    if len(rewrite_candidates) > 1:
        raise ValueError(
            f"More than one possible rewrite column in {path}: {rewrite_candidates}. "
            f"Please drop unused columns or hardcode the rewrite column."
        )

    return rewrite_candidates[0]


def add_toxicity_columns(df: pd.DataFrame, orig_col: str, rewrite_clean_col: str) -> pd.DataFrame:
    print("  - Computing toxicity scores...")
    orig_texts = df[orig_col].fillna("").astype(str).tolist()
    new_texts = df[rewrite_clean_col].fillna("").astype(str).tolist()

    tox_before = tox_model.predict(orig_texts)["toxicity"]
    tox_after = tox_model.predict(new_texts)["toxicity"]

    df["toxicity_before"] = tox_before
    df["toxicity_after"] = tox_after
    df["toxicity_reduction"] = df["toxicity_before"] - df["toxicity_after"]
    return df


def add_similarity_column(df: pd.DataFrame, orig_col: str, rewrite_clean_col: str) -> pd.DataFrame:
    print("  - Computing semantic similarity...")

    orig_texts = df[orig_col].fillna("").astype(str).tolist()
    new_texts = df[rewrite_clean_col].fillna("").astype(str).tolist()

    emb_orig = sim_model.encode(orig_texts, convert_to_tensor=True, show_progress_bar=True)
    emb_new = sim_model.encode(new_texts, convert_to_tensor=True, show_progress_bar=True)

    cos_sim_diag = util.cos_sim(emb_orig, emb_new).diagonal()
    df["semantic_similarity"] = cos_sim_diag.cpu().numpy()
    return df


def summarize_metrics(df: pd.DataFrame, model_name: str) -> dict:
    return {
        "model": model_name,
        "mean_toxicity_before": df["toxicity_before"].mean(),
        "mean_toxicity_after": df["toxicity_after"].mean(),
        "mean_toxicity_reduction": df["toxicity_reduction"].mean(),
        "mean_semantic_similarity": df["semantic_similarity"].mean(),
        "std_semantic_similarity": df["semantic_similarity"].std(),
        "n_examples": len(df),
    }


def make_manual_review_csv(df: pd.DataFrame, model_name: str, orig_col: str,
                           rewrite_clean_col: str, max_samples: int = 100):
    print("  - Creating manual review file for contextual coherence...")

    if len(df) > max_samples:
        review_df = df.sample(max_samples, random_state=42).copy()
    else:
        review_df = df.copy()

    cols = []
    if ID_COL in review_df.columns:
        cols.append(ID_COL)
    if IMG_COL in review_df.columns:
        cols.append(IMG_COL)

    cols += [orig_col, rewrite_clean_col]
    review_df = review_df[cols].copy()

    # Rename for annotators
    review_df = review_df.rename(columns={
        orig_col: "original_text",
        rewrite_clean_col: "rewritten_text",
    })

    review_df["rate_coherence_1_5"] = ""
    review_df["rate_relevance_1_5"] = ""
    review_df["rate_positive_tone_1_5"] = ""
    review_df["comments"] = ""

    out_path = f"eval/manual_review_{model_name}.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    review_df.to_csv(out_path, index=False)
    print(f"    -> Saved manual review template to {out_path}")


# -----------------------------
# Main
# -----------------------------
def main():
    all_summaries = []

    for model_name, path in INPUT_FILES.items():
        if not os.path.exists(path):
            print(f"\n=== Skipping {model_name}: file not found ({path}) ===")
            continue

        print(f"\n=== Evaluating {model_name} ({path}) ===")

        df = pd.read_csv(path)

        if ORIG_COL not in df.columns:
            raise ValueError(
                f"Expected original text column '{ORIG_COL}' in {path}. "
                f"Found columns: {list(df.columns)}"
            )

        rewrite_col = find_rewrite_column(df, path)
        print(f"  - Detected rewrite column: {rewrite_col}")

        # Clean the rewrite column (handles Safe_Text / Safe_Image or plain text)
        df["rewrite_clean"] = clean_rewrite_column(df, rewrite_col)

        # Compute metrics
        df = add_toxicity_columns(df, ORIG_COL, "rewrite_clean")
        df = add_similarity_column(df, ORIG_COL, "rewrite_clean")

        # Save per row metrics
        os.makedirs("results", exist_ok=True)
        out_csv = f"results/{model_name}_evaluated.csv"
        df.to_csv(out_csv, index=False)
        print(f"  -> Saved per example metrics to {out_csv}")

        # Summary row
        summary = summarize_metrics(df, model_name)
        all_summaries.append(summary)

        # Manual review CSV
        make_manual_review_csv(df, model_name, ORIG_COL, "rewrite_clean", max_samples=100)

    # Global summary
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        os.makedirs("eval", exist_ok=True)
        summary_path = "eval/model_metrics_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        print("\n=== Overall summary ===")
        print(summary_df)
        print(f"\n-> Saved summary to {summary_path}")

        # Extra: compare Gemini vs Gemini few-shot if both are present
        if {"gemini", "gemini_fewshot"}.issubset(set(summary_df["model"])):
            print("\n=== Gemini vs Gemini Few-Shot Comparison ===")
            reg = summary_df[summary_df["model"] == "gemini"].iloc[0]
            fs = summary_df[summary_df["model"] == "gemini_fewshot"].iloc[0]

            comparison = pd.DataFrame([
                {
                    "metric": "mean_toxicity_before",
                    "gemini": reg["mean_toxicity_before"],
                    "gemini_fewshot": fs["mean_toxicity_before"],
                },
                {
                    "metric": "mean_toxicity_after",
                    "gemini": reg["mean_toxicity_after"],
                    "gemini_fewshot": fs["mean_toxicity_after"],
                },
                {
                    "metric": "mean_toxicity_reduction",
                    "gemini": reg["mean_toxicity_reduction"],
                    "gemini_fewshot": fs["mean_toxicity_reduction"],
                },
                {
                    "metric": "mean_semantic_similarity",
                    "gemini": reg["mean_semantic_similarity"],
                    "gemini_fewshot": fs["mean_semantic_similarity"],
                },
            ])
            print(comparison.to_string(index=False))
            comparison.to_csv("eval/model_metrics_summary_v2.csv", index=False)
            print("\n-> Saved Gemini vs Gemini few-shot comparison to eval/gemini_vs_fewshot_summary.csv")


if __name__ == "__main__":
    main()

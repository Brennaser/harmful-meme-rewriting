import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse


# ===============================================
# COMPARISON SCRIPT
# ===============================================

def load_results(chatgpt_csv, claude_csv):
    """Load result CSVs for ChatGPT and Claude"""
    results = {}

    if chatgpt_csv:
        results['ChatGPT'] = pd.read_csv(chatgpt_csv)
    if claude_csv:
        results['Claude'] = pd.read_csv(claude_csv)

    return results


def calculate_rewrite_distance(original, rewrite):
    """
    Calculate how much the text changed (0-1 scale)
    Higher = more changed
    """
    if not rewrite or pd.isna(rewrite) or rewrite == "":
        return 0

    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 2))
    try:
        vectors = vectorizer.fit_transform([str(original), str(rewrite)])
        similarity = cosine_similarity(vectors)[0, 1]
        distance = 1 - similarity
        return distance
    except:
        return 0


def analyze_model_output(df, model_name, rewrite_column):
    """Analyze a single model's output"""
    print(f"\n{'=' * 60}")
    print(f"Analysis for: {model_name}")
    print(f"{'=' * 60}")

    # Count empty rewrites
    empty_count = df[rewrite_column].isna().sum() + (df[rewrite_column] == "").sum()
    total_count = len(df)
    completion_rate = (total_count - empty_count) / total_count * 100

    print(f"Completion Rate: {completion_rate:.1f}% ({total_count - empty_count}/{total_count})")

    # Calculate rewrite distances
    distances = []
    for idx, row in df.iterrows():
        original = row['text']
        rewrite = row[rewrite_column]
        if pd.notna(rewrite) and rewrite != "":
            distance = calculate_rewrite_distance(original, rewrite)
            distances.append(distance)

    if distances:
        avg_distance = np.mean(distances)
        print(f"Average Rewrite Distance: {avg_distance:.3f} (0=identical, 1=completely different)")
        print(f"  Min: {np.min(distances):.3f}, Max: {np.max(distances):.3f}")

    # Length analysis
    lengths = []
    for idx, row in df.iterrows():
        rewrite = row[rewrite_column]
        if pd.notna(rewrite) and rewrite != "":
            lengths.append(len(str(rewrite)))

    if lengths:
        avg_length = np.mean(lengths)
        print(f"Average Rewrite Length: {avg_length:.0f} characters")
        print(f"  Min: {np.min(lengths)}, Max: {np.max(lengths)}")

    return {
        'model': model_name,
        'completion_rate': completion_rate,
        'avg_distance': avg_distance if distances else 0,
        'avg_length': avg_length if lengths else 0,
        'total': total_count,
        'completed': total_count - empty_count
    }


def show_sample_comparisons(results, num_samples=5):
    """Show sample rewrites from each model side-by-side"""
    print(f"\n{'=' * 60}")
    print(f"Sample Comparisons (First {num_samples} memes)")
    print(f"{'=' * 60}\n")

    # Get first dataframe for original text
    first_df = list(results.values())[0]

    for idx in range(min(num_samples, len(first_df))):
        row = first_df.iloc[idx]
        print(f"Row {idx}: {row['text'][:60]}...")
        print("-" * 80)

        for model_name, df in results.items():
            rewrite_col = [col for col in df.columns if 'rewrite' in col][0]
            rewrite = df.iloc[idx][rewrite_col]
            if pd.isna(rewrite):
                rewrite = "[EMPTY]"
            print(f"{model_name:12} | {str(rewrite)[:70]}")
        print()


def comparison_summary(analyses):
    """Create summary comparison table"""
    print(f"\n{'=' * 60}")
    print("SUMMARY COMPARISON")
    print(f"{'=' * 60}\n")

    # Create dataframe
    summary_df = pd.DataFrame(analyses)

    print(summary_df.to_string(index=False))
    print()

    # Rankings
    print("RANKINGS:")
    print(f"  Best Completion Rate: {summary_df.loc[summary_df['completion_rate'].idxmax(), 'model']}")
    print(f"  Most Changed Text: {summary_df.loc[summary_df['avg_distance'].idxmax(), 'model']}")
    print(f"  Longest Rewrites: {summary_df.loc[summary_df['avg_length'].idxmax(), 'model']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chatgpt", default="results/memes_chatgpt.csv", help="ChatGPT results CSV")
    parser.add_argument("--claude", default="results/memes_claude.csv", help="Claude results CSV")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to show")
    args = parser.parse_args()

    # Load results
    results = load_results(args.chatgpt, args.claude)

    if not results:
        print("No results found. Please provide CSV files.")
        return

    print("\n" + "=" * 60)
    print("CHATGPT vs CLAUDE COMPARISON")
    print("=" * 60)

    # Analyze each model
    analyses = []

    for model_name, df in results.items():
        rewrite_col = [col for col in df.columns if 'rewrite' in col][0]
        analysis = analyze_model_output(df, model_name, rewrite_col)
        analyses.append(analysis)

    # Show samples
    show_sample_comparisons(results, args.samples)

    # Summary
    comparison_summary(analyses)


if __name__ == "__main__":
    main()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_toxicity_plot_simple(file_path):
    """
    Reads model metrics from a CSV file (now assuming comma-separated) 
    and generates a grouped bar chart comparing mean toxicity before and after.

    Args:
        file_path (str): The full path to the input CSV file.
    """
    try:
        # --- CHANGE MADE HERE ---
        # Changed the separator from whitespace (\\s+) to comma (,) 
        # based on the column headers you provided.
        df = pd.read_csv(file_path, sep=",")
        print(f"Successfully loaded data from: {file_path}")

    except FileNotFoundError:
        print(f"Error: File not found at path: {file_path}")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    # Check for required columns
    required_cols = ['model', 'mean_toxicity_before', 'mean_toxicity_after']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: The CSV must contain the columns: {required_cols}")
        print(f"Found columns: {list(df.columns)}")
        return

    # Prepare data for plotting (Melting to long format)
    df_melted = df.melt(
        id_vars='model',
        value_vars=['mean_toxicity_before', 'mean_toxicity_after'],
        var_name='Toxicity_Type',
        value_name='Mean_Toxicity'
    )

    df_melted['Toxicity_Type'] = df_melted['Toxicity_Type'].replace({
        'mean_toxicity_before': 'Before Intervention',
        'mean_toxicity_after': 'After Intervention'
    })

    # Create the Grouped Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.35
    models = df['model'].unique()
    x = np.arange(len(models))  # the label locations

    # Plot 'Before' bars
    ax.bar(
        x - bar_width/2, 
        df_melted[df_melted['Toxicity_Type'] == 'Before Intervention']['Mean_Toxicity'],
        bar_width, 
        label='Before Intervention', 
        color='#1f77b4',
        edgecolor='black'
    )

    # Plot 'After' bars
    ax.bar(
        x + bar_width/2, 
        df_melted[df_melted['Toxicity_Type'] == 'After Intervention']['Mean_Toxicity'],
        bar_width, 
        label='After Intervention', 
        color='#ff7f0e',
        edgecolor='black'
    )

    # Styling and Labels
    ax.set_ylabel('Mean Toxicity Score', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_title('Mean Toxicity Before and After Intervention by Model', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.set_ylim(0, df['mean_toxicity_before'].max() * 1.1)
    
    plt.tight_layout()

    # Save the plot
    output_filename = 'mean_toxicity_comparison_simple.png'
    plt.savefig(output_filename)
    print(f"\nPlot saved successfully as: {output_filename}")


# --- Execution ---
# Define the path exactly as you requested
file_path_to_use = "/home/kiwi-pandas/Documents/harmful-meme-rewriting/eval/model_metrics_summary.csv"

# Call the function
generate_toxicity_plot_simple(file_path_to_use)
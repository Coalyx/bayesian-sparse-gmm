import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set "Nature-figure style" parameters
def set_nature_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'legend.title_fontsize': 9,
        'lines.linewidth': 1.5,
        'axes.linewidth': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.format': 'png',
        'savefig.bbox': 'tight'
    })

# Professional colorblind-friendly palettes
MODEL_COLORS = {
    "K-Means": "#1f77b4",          # Blue
    "DBSCAN": "#ff7f0e",           # Orange
    "GMM": "#2ca02c",              # Green
    "BayesianSparseGMM": "#d62728", # Red
    "BayesianSparseGMM (SVI)": "#9467bd" # Purple
}

def plot_metric_comparisons(df: pd.DataFrame, output_dir: str):
    """
    Plots bar charts for ARI, NMI, and MCR comparisons across models.
    Groups by Dataset Category.
    """
    set_nature_style()
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = ["ARI", "NMI", "MCR"]
    y_labels = {"ARI": "Adjusted Rand Index", "NMI": "Normalized Mutual Information", "MCR": "Mis-Clustering Rate"}
    
    for metric in metrics:
        if metric not in df.columns:
            continue
            
        plt.figure(figsize=(10, 5))
        
        # Use seaborn barplot
        ax = sns.barplot(
            data=df, 
            x="Dataset", 
            y=metric, 
            hue="Model", 
            palette=MODEL_COLORS,
            errorbar=None
        )
        
        plt.title(f"Clustering Performance: {y_labels[metric]}")
        plt.ylabel(y_labels[metric])
        plt.xlabel("Dataset")
        plt.xticks(rotation=45, ha='right')
        
        # Adjust legend
        plt.legend(title="Algorithm", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}_comparison.png"))
        plt.close()

def plot_scalability(df: pd.DataFrame, output_dir: str):
    """
    Plots line/scatter plots for Time and Memory vs N and D.
    """
    set_nature_style()
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort by N and D to create smooth lines
    df_n = df.sort_values("N")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Execution Time vs N
    sns.lineplot(
        data=df_n, 
        x="N", 
        y="Time_s", 
        hue="Model", 
        marker="o", 
        palette=MODEL_COLORS, 
        ax=axes[0]
    )
    axes[0].set_title("Execution Time vs Dataset Size")
    axes[0].set_ylabel("Time (seconds)")
    axes[0].set_xlabel("Number of Samples ($N$)")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False)
    
    # Memory Consumption vs N
    sns.lineplot(
        data=df_n, 
        x="N", 
        y="Memory_MB", 
        hue="Model", 
        marker="o", 
        palette=MODEL_COLORS, 
        ax=axes[1]
    )
    axes[1].set_title("Peak Memory Usage vs Dataset Size")
    axes[1].set_ylabel("Peak Memory (MB)")
    axes[1].set_xlabel("Number of Samples ($N$)")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].legend(frameon=False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scalability_analysis.png"))
    plt.close()

def generate_all_plots(csv_path: str, output_dir: str):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    print(f"Generating Nature-style plots in {output_dir}...")
    plot_metric_comparisons(df, output_dir)
    plot_scalability(df, output_dir)
    print("Plot generation complete.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "results/benchmark_results.csv"
        
    out_dir = "visualize"
    generate_all_plots(csv_path, out_dir)

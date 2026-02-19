import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def box_plot(df : pd.DataFrame, title : str, type_cancer : str) -> None:
    """
    Function to make box plots
    """
    if type_cancer == "Luminal A":
        df = df[df["Tumor-Cancer"] == "Luminal A"].iloc[:, 1:]
    elif type_cancer == "Luminal B":
        df = df[df["Tumor-Cancer"] == "Luminal B"].iloc[:, 1:]
    
    plt.figure(figsize=(12,5))
    plt.boxplot(np.log2(df[1:21].T + 1))  # transpose
    plt.title(title, fontsize=14)
    plt.xticks(rotation=90)
    plt.show()
    

def histogram_log2(df: pd.DataFrame, title: str, type_cancer : str) -> None:
    if type_cancer == "Luminal A":
        df = df[df["Tumor-Cancer"] == "Luminal A"].iloc[:, 1:]
    elif type_cancer == "Luminal B":
        df = df[df["Tumor-Cancer"] == "Luminal B"].iloc[:, 1:]
        
    x = np.log2(df.iloc[1:20000].T.values.flatten() + 1)
    counts, bins = np.histogram(x, bins=100)
    counts_percentage = (counts / counts.sum()) * 100
    bin_width = bins[1] - bins[0]

    plt.figure(figsize=(12,5))

    plt.bar(
        bins[:-1],
        counts_percentage,
        width=bin_width,
        edgecolor="black",
        alpha=0.85,
        align="edge"
    )
    plt.title(
        f"Distribution of log2 {type_cancer} Values Before Filtering",
        fontsize=14,
        fontweight="bold"
    )

    plt.xlabel("log2(count + 1)", fontsize=12)
    plt.ylabel("Percentage of observations (%)", fontsize=12)

    plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.xlim(0, np.max(bins))
    plt.tight_layout()

    plt.show()
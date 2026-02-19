import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def box_plot(df : pd.DataFrame, title : str, type_cancer : str) -> None:
    """
    Function to make box plots
    """
    if type_cancer == "Luminal A":
        df = df[df["Tumor-Cancer"] == type_cancer] 
    elif type_cancer == "Luminal B":
        df = df[df["Tumor-Cancer"] == type_cancer] 
    plt.figure(figsize=(12,5))
    plt.boxplot(np.log2(df + 1))  # transpose
    plt.title(title, fontsize=14)
    plt.xticks(rotation=90)
    plt.show()
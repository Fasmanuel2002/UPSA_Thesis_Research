import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from sklearn.decomposition import PCA
from typing import List, Optional, Any


def box_plot(data : Any, title : str, type_cancer : Optional[str],
             range_min : int, range_max : int ) -> None:
    """
    Function to make box plots
    """
    if isinstance(data, pd.DataFrame):
        if type_cancer is not None:
            valid_types_cancer = ["Luminal A", "Luminal B", "TNBC", "HER2-enriched"]
            if type_cancer not in valid_types_cancer:
                raise ValueError(f"Error in the valid types, needs to be {valid_types_cancer}")
        
        data = data[data["Tumor-Cancer"] == type_cancer].iloc[: ,1:] 
        data = np.log2(data[range_min:range_max].T + 1)
        
    elif isinstance(data, np.ndarray):
        data = data[range_min:range_max].T 
    
    plt.figure(figsize=(12,5))
    plt.boxplot(data)  # transpose
    plt.title(title, fontsize=14)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


    

def histogram_log2(df: pd.DataFrame, title: str, type_cancer : str) -> None:
    
    if type_cancer is not None:
            valid_types_cancer = ["Luminal A", "Luminal B", "TNBC", "HER2-enriched"]
            if type_cancer not in valid_types_cancer:
                raise ValueError(f"Error in the valid types, needs to be {valid_types_cancer}")
    
    df = df[df["Tumor-Cancer"] == type_cancer].iloc[: 1:]
    
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
    

def PCA_2_variables(df : pd.DataFrame,  cancer_type_one : str, cancer_type_two : str) -> None:
    """
    Function to make the plot to compare only two type of cancer mamals
    Types:
        - Luminal A
        - Luminal B
        - Triple Negative (TNBC)
        - HER-enriched 
    """
    y = df["Tumor-Cancer"]
    X = df.drop(columns=["Tumor-Cancer"])


    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2, random_state=0)
    components = pca.fit_transform(X_scaled)


    df_plot = pd.DataFrame(components, columns=["PC1", "PC2"])
    df_plot["Tumor-Cancer"] = y.values
    df_plot["Patient_ID"] = df.index

    labels = {
        "PC1": f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
        "PC2": f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
    }


    fig = px.scatter(
        df_plot,
        x="PC1",
        y="PC2",
        color="Tumor-Cancer",
        labels=labels,
        hover_name="Patient_ID",
        title=f"PCA de Subtypes of cancer of mama of {cancer_type_one} and {cancer_type_two}",
        opacity=0.7,
        color_discrete_map={f"{cancer_type_one}": "#1f77b4", f"{cancer_type_two}": "#ec3204"},
    )

    fig.update_traces(marker=dict(size=6))
    fig.show()


def PCA_variables_log2(df : pd.DataFrame,  cancer_types : List[str]) -> None:
    """
    Function to make the plot to compare only two type of cancer mamals
    Types:
        - Luminal A
        - Luminal B
        - Triple Negative (TNBC)
        - HER-enriched 
    """
    y = df["Tumor-Cancer"]
    X = df.drop(columns=["Tumor-Cancer"])


    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)


    X_log = np.log2(X +1)
    X_scaled = StandardScaler().fit_transform(X_log)

    pca = PCA(n_components=2, random_state=0)
    components = pca.fit_transform(X_scaled)


    df_plot = pd.DataFrame(components, columns=["PC1", "PC2"])
    df_plot["Tumor-Cancer"] = y.values
    df_plot["Patient_ID"] = df.index

    labels = {
        "PC1": f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
        "PC2": f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
    }


    fig = px.scatter(
        df_plot,
        x="PC1",
        y="PC2",
        color="Tumor-Cancer",
        labels=labels,
        hover_name="Patient_ID",
        title=f"PCA de Subtypes of cancer of mama with Log2 {cancer_types[0]}, {cancer_types[1]}, {cancer_types[2]},{cancer_types[3]} ",
        opacity=0.7,
        color_discrete_map={f"{cancer_types[0]}": "#1f77b4",
                            f"{cancer_types[1]}": "#ec3204",
                            f"{cancer_types[2]}": "#49fa09",
                            f"{cancer_types[3]}": "#fa09fa"},
    )

    fig.update_traces(marker=dict(size=6))
    fig.show()



def PCA_4_scatter_matrix_log2(df : pd.DataFrame,  cancer_types : List[str]) -> None:
    """
    Function to make the plot to compare only four type of cancer mamals
    Types:
        - Luminal A
        - Luminal B
        - Triple Negative (TNBC)
        - HER-enriched 
    """
    y = df["Tumor-Cancer"]
    X = df.drop(columns=["Tumor-Cancer"])


    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)


    X_log = np.log2(X +1)
    X_scaled = StandardScaler().fit_transform(X_log)

    pca = PCA(n_components=4, random_state=0)
    components = pca.fit_transform(X_scaled)


    df_plot = pd.DataFrame(components, columns=["PC1", "PC2", "PC3", "PC4"])
    df_plot["Tumor-Cancer"] = y.values
    df_plot["Patient_ID"] = df.index

    labels = {
        "PC1": f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
        "PC2": f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
        "PC3": f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)",
        "PC4": f"PC4 ({pca.explained_variance_ratio_[3]*100:.1f}%)",
    }


    fig = px.scatter_matrix(
        df_plot,
        dimensions=["PC1", "PC2", "PC3", "PC4"],
        color="Tumor-Cancer",
        labels=labels,
        hover_name="Patient_ID",
        title=f"PCA de Subtypes of cancer of mama with Log2 {cancer_types[0]}, {cancer_types[1]}, {cancer_types[2]}, {cancer_types[3]}",
        opacity=0.7,
        color_discrete_map={f"{cancer_types[0]}": "#1f77b4",
                            f"{cancer_types[1]}": "#ec3204",
                            f"{cancer_types[2]}": "#49fa09",
                            f"{cancer_types[3]}": "#fa09fa"},
    )

    fig.update_traces(marker=dict(size=6))
    fig.show()


    
    
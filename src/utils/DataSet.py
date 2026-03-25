import pandas as pd
from typing import Tuple
from .Preprocessing import Preprocessor
import numpy as np
from sksurv.util import Surv


pp = Preprocessor()

def split_data_time_months(df_mRNA: pd.DataFrame, 
                         df_clinical: pd.DataFrame,
                         gene: str,
                         time : int) -> Tuple:
    
    df_single_gene = pp.gene_to_long(df_mRNA, gene)
    
    df_gene_merged = df_single_gene.merge(df_clinical, on="Sample ID", how="inner")
    
    df_gene_merged["Overall Survival (Months)"] = pd.to_numeric(
        df_gene_merged["Overall Survival (Months)"], errors="coerce"
    )
    
    status = df_gene_merged["Overall Survival Status"].astype(str).str.strip()
    df_gene_merged["event"] = status.str.contains("DECEASED", na=False)
    
    df_gene_merged["event_60"] = df_gene_merged["event"].copy()
    df_gene_merged["time_60"] = np.minimum(df_gene_merged["Overall Survival (Months)"], time)
    
    df_gene_merged.loc[
        df_gene_merged["Overall Survival (Months)"] > time, "event_60"
    ] = False
    
    df_gene_merged["expression"] = pd.to_numeric(
        df_gene_merged["expression"], errors="coerce"
    )
    
    df_gene_merged["expression"] = np.log2(df_gene_merged["expression"] +   1)
    
    df_gene_merged = df_gene_merged.dropna(subset=["time_60", "expression"]).copy()
    
    X = df_gene_merged[["expression"]]
    
    Y_surv = Surv.from_dataframe(
        event="event_60",
        time="time_60",
        data=df_gene_merged
    )
    
    return X, Y_surv, df_gene_merged
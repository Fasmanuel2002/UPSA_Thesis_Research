
from typing import Tuple

import numpy as np
import pandas as pd
from src.utils.Preprocessing import Preprocessor
from numpy import ndarray
from sklearn.preprocessing import StandardScaler



class TorchPreprocessing:
    def __init__(self, df_mrna_clean : pd.DataFrame, df_clinical_data_clean : pd.DataFrame, number_columns : int) -> None:
        self.pp = Preprocessor()
        self.genes_expression = self.genes_expression = (df_mrna_clean["Hugo_Symbol"].drop_duplicates().sample(number_columns, random_state=42).tolist())
        self.df_mrna_clean = df_mrna_clean
        self.df_clinical_data_clean = df_clinical_data_clean
        
    def get_expression_cols(self) -> pd.DataFrame:
        expressions_genes_cols = pd.DataFrame()
        for index, gene in enumerate(self.genes_expression):
            df_genes = self.pp.gene_to_long(self.df_mrna_clean, gene)
            if index == 0:
                expressions_genes_cols = (df_genes[["Sample ID", "expression"]]
                                         .rename(columns={"expression": gene}).copy())
            else: 
                expressions_genes_cols = expressions_genes_cols.merge(
                df_genes[["Sample ID", "expression"]].rename(columns={"expression": gene}),
                on="Sample ID",
                how="inner"
            )      
        return expressions_genes_cols
    
    def get_comparation_df(self) -> pd.DataFrame:
        expression_genes_cols = self.get_expression_cols()
        df_merged = pd.merge(
            left=self.df_clinical_data_clean, 
            right=expression_genes_cols,
            left_on="Sample ID",
            right_on="Sample ID",
        )
        
        cols = ["Sample ID","Tumor-Cancer", "Overall Survival Status", "Overall Survival (Months)"] + list(self.genes_expression)
        comparation_df = df_merged.loc[
            df_merged["Tumor-Cancer"].isin(["Luminal A", "Luminal B", "TNBC", "HER2-enriched"]),
            cols
        ]
        return self.pp.eliminate_zero_genes(comparation_df, "Tumor-Cancer", threshold=0.8)
    
    
    def get_data_set(self, time_months : int) -> Tuple:
        comparation_df = self.get_comparation_df()
        status = comparation_df["Overall Survival Status"].astype(str).str.strip()
        
        comparation_df["event"] = status.str.contains("DECEASED", na=False)
        
        comparation_df["event_60"] = comparation_df["event"].copy()
        
        comparation_df["time_60"] =  np.minimum(comparation_df["Overall Survival (Months)"], time_months)
        comparation_df.loc[
            comparation_df["Overall Survival (Months)"] > time_months, "event_60"
        ] = False

        comparation_df = comparation_df.dropna(subset=["time_60"]).copy()
        
        comparation_df = comparation_df.drop(columns="Sample ID")


        valid_genes = [g for g in self.genes_expression if g in comparation_df.columns]

        X = comparation_df.loc[:, valid_genes].astype(float)
        
        scaler = StandardScaler()
        
        X_scaled = scaler.fit_transform(X)
        
        durations = comparation_df["time_60"].values
        
        events = comparation_df["event_60"].values.astype(float)
        
        return X_scaled, durations, events, scaler

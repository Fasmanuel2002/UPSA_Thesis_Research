from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.utils.Preprocessing import Preprocessor


class TorchPreprocessing:
    def __init__(
        self, df_mrna_clean: pd.DataFrame,df_clinical_data_clean: pd.DataFrame,number_columns: int) -> None:
        self.pp = Preprocessor()

        unique_genes = df_mrna_clean["Hugo_Symbol"].drop_duplicates()
        self.genes_expression = unique_genes.sample(min(number_columns, len(unique_genes)),random_state=42 ).tolist()
        self.df_mrna_clean = df_mrna_clean
        self.df_clinical_data_clean = df_clinical_data_clean

    def get_expression_cols(self) -> pd.DataFrame:
        expr = self.df_mrna_clean[self.df_mrna_clean["Hugo_Symbol"].isin(self.genes_expression)].copy()

        expr = expr.drop_duplicates(subset="Hugo_Symbol")

        expr = expr.set_index("Hugo_Symbol")

        valid_sample_ids = set(self.df_clinical_data_clean["Sample ID"].astype(str))
        sample_cols = [col for col in expr.columns if str(col) in valid_sample_ids]

        expr = expr[sample_cols]

        expr = expr.T
        expr.index.name = "Sample ID"
        expr = expr.reset_index()

        return expr

    def get_comparation_df(self) -> pd.DataFrame:
        expression_genes_cols = self.get_expression_cols()

        df_merged = pd.merge(self.df_clinical_data_clean, expression_genes_cols,on="Sample ID",how="inner")

        tumor_types = {"Luminal A", "Luminal B", "TNBC", "HER2-enriched"}

        cols = ["Sample ID","Tumor-Cancer","Overall Survival Status","Overall Survival (Months)"] + [gene for gene in self.genes_expression if gene in df_merged.columns]

        comparation_df = df_merged.loc[df_merged["Tumor-Cancer"].isin(tumor_types),cols].copy()

        return self.pp.eliminate_zero_genes(comparation_df,"Tumor-Cancer",threshold=0.8)

    def get_data_set(self, time_months: int) -> Tuple:
        comparation_df = self.get_comparation_df().copy()

        status = comparation_df["Overall Survival Status"].astype(str).str.strip()
        comparation_df["event"] = status.str.contains("DECEASED", na=False)

        comparation_df["time_60"] = np.minimum(
            comparation_df["Overall Survival (Months)"],
            time_months
        )

        comparation_df["event_60"] = comparation_df["event"]
        comparation_df.loc[
            comparation_df["Overall Survival (Months)"] > time_months,
            "event_60"
        ] = False

        comparation_df = comparation_df.dropna(subset=["time_60"]).copy()
        comparation_df = comparation_df.drop(columns=["Sample ID"])

        valid_genes = [g for g in self.genes_expression if g in comparation_df.columns]

        X = comparation_df[valid_genes].astype(float)
        durations = comparation_df["time_60"].to_numpy()
        events = comparation_df["event_60"].astype(float).to_numpy()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return X_scaled, durations, events, scaler
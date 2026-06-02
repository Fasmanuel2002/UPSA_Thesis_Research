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

    def get_data_set(self, time_months: int, return_ids: bool = False,
                     apply_scaler: bool = True) -> Tuple:
        """Build the (X, durations, events, scaler[, ids]) tuple for modelling.

        ``apply_scaler`` (default True for backward compatibility) fits a
        StandardScaler on the FULL dataset and returns the transformed X.
        That introduces test-set leakage because val/test contribute to the
        feature means and SDs used at train time. The methodologically
        correct usage is ``apply_scaler=False``: this returns the raw X
        (and an UNFITTED StandardScaler) so the caller can split first and
        then fit the scaler on the training fold only:

            X, durs, evs, scaler, ids = tp.get_data_set(60, return_ids=True,
                                                         apply_scaler=False)
            # split by IDs ...
            X_train = scaler.fit_transform(X[train_mask])
            X_val   = scaler.transform(X[val_mask])
            X_test  = scaler.transform(X[test_mask])
        """
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
        sample_ids = comparation_df["Sample ID"].to_numpy()
        comparation_df = comparation_df.drop(columns=["Sample ID"])

        valid_genes = [g for g in self.genes_expression if g in comparation_df.columns]

        X = comparation_df[valid_genes].astype(float)
        durations = comparation_df["time_60"].to_numpy()
        events = comparation_df["event_60"].astype(float).to_numpy()

        scaler = StandardScaler()
        if apply_scaler:
            X_out = scaler.fit_transform(X)
        else:
            X_out = X.to_numpy(dtype=float)  # caller must fit scaler on train fold

        if return_ids:
            return X_out, durations, events, scaler, sample_ids
        return X_out, durations, events, scaler
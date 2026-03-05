import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from inmoose import limma
import numpy as np
from typing import Tuple


KEEP_COLUMNS = [   
    "Sample ID", "Patient ID",
    "Overall Survival (Months)", "Overall Survival Status",
    "Diagnosis Age",
    "Neoplasm Disease Stage American Joint Committee on Cancer Code",
    "Lymph Node(s) Examined Number",
    "Menopause Status",
    "ER Status By IHC",
    "PR status by ihc",
    "HER2 fish status",
    "HER2 ihc score",
]


class Preprocessor:
    def __init__(self) -> None:
        pass

    
    def classify_cancer_type(self, df_clinical_data : pd.DataFrame) -> list:
        """
        Function to classify between the different type of Cancers,
        Returns: Luminal A (ER+ HER2-) 
                Luminal B (ER+ HER2+)
                HER-enriched (ER- PR- HER2+ )
                TNBC (ER- PR- HER2- )
                
        """
        res = []
        
        for _, row in df_clinical_data.iterrows():
            Her2_fish_status = row.get("HER2 fish status", pd.NA)
            Her2_ihc = row.get("HER2 ihc score", pd.NA)
            ER_status = row.get("ER Status By IHC", pd.NA)
            PR_status = row.get("PR status by ihc", pd.NA)
            
            if pd.notna(Her2_fish_status):
                if (ER_status == "Positive") and (Her2_fish_status == "Negative"):
                    res.append("Luminal A")
                    
                elif (ER_status == "Positive") and (Her2_fish_status == "Positive"):
                    res.append("Luminal B")
                    
                elif (ER_status == "Negative") and (PR_status == "Negative") and (Her2_fish_status == "Positive"):
                    res.append("HER2-enriched")
                    
                elif (ER_status == "Negative") and (PR_status == "Negative") and (Her2_fish_status == "Negative"):
                    res.append("TNBC")
                
                else:
                    res.append("<UNK>")
                    
                continue
            
            if pd.isna(Her2_ihc):
                res.append("<UNK>")
                
                continue
                    
            if (Her2_ihc == 2):
                res.append("<UNK>")
                        
            elif (ER_status == "Positive") and (Her2_ihc <= 1):
                res.append("Luminal A")
                        
            elif (ER_status == "Positive") and (Her2_ihc >= 3):
                res.append("Luminal B")
                    
            elif (ER_status == "Negative") and (PR_status == "Negative") and (Her2_ihc >= 3):
                res.append("HER2-enriched")
                    
            elif (ER_status == "Negative") and (PR_status == "Negative") and (Her2_ihc <= 1):
                res.append("TNBC")
                    
            else:
                res.append("<UNK>")
                
        return res



    def elimnation_zeros(self, df : pd.DataFrame, column : str) -> pd.DataFrame:  
        genes = df.columns[1:]
        
        zeros_genes = (df[genes] == 0).sum(axis=0)
        
        max_number_of_zeros = zeros_genes.max()
        
        avg_number_of_zeros = zeros_genes.mean()
        
        median_number_of_zeros = zeros_genes.median()
        
        min_number_of_zeros = zeros_genes.min()
        
        print(f"Max of zeros per row in the dataset: {max_number_of_zeros}")
        
        print(f"Avg of zeros per row in the dataset: {avg_number_of_zeros}")
        
        print(f"Median of zeros per row in the dataset: {median_number_of_zeros}")
        
        print(f"Min of zeros per row in the dataset: {min_number_of_zeros}")
        
        keep_columns = zeros_genes <= avg_number_of_zeros 
        
        df_return = pd.concat(
            [df[column], df.loc[:, genes[keep_columns]]],
            axis=1
        )
        
        print(f"After the 0 elimination: {df_return.shape[1]}") 
        
        return df_return
    
    def clean_columns_dataset(self, df : pd.DataFrame) -> pd.DataFrame:
        """
        Function for cleaning columns that doesn't value for the clinical dataset
        """
        return df[[column for column in KEEP_COLUMNS if column in df.columns]].copy()


    def initialize_DeseqDataSet(self, counts_data : pd.DataFrame, 
                        metadata:pd.DataFrame, 
                        design : str) -> DeseqDataSet:
        """
        For initializing the DeseqDataSet variable
        """
        inference = DefaultInference(n_cpus=2)
        
        deseqDataSet = DeseqDataSet(
            counts=counts_data,
            metadata=metadata,
            design=f'~{design}',
            inference=inference
        )
        
        return deseqDataSet


    def total_type_len_type_cancer(self, df: pd.DataFrame) -> list:
        
        list_df = self.classify_cancer_type(df_clinical_data=df)

        luminal_A = [x for x in list_df if x == "Luminal A"]
        luminal_B = [x for x in list_df if x == "Luminal B"]
        HER2_enriched = [x for x in list_df if x == "HER2-enriched"]
        TNBC = [x for x in list_df if x == "TNBC"]
        UNK = [x for x in list_df if x == "<UNK>"]

        print(f"Luminal A: {len(luminal_A)} - Total(%): {len(luminal_A) / len(df):.2f}")
        print(f"Luminal B: {len(luminal_B)} - Total(%):{len(luminal_B) / len(df):.2f}")
        print(f"HER2-enriched: {len(HER2_enriched)} - Total(%):{len(HER2_enriched) / len(df):.2f}")
        print(f"TNBC: {len(TNBC)} - Total(%){len(TNBC) / len(df):.2f} ")
        print(f"UNK: {len(UNK)} - Total(%) {len(UNK) / len(df):.2f}")

        return list_df

    
    def merge_datasets(self, df_clinical_data : pd.DataFrame ,
                       df_mRNA : pd.DataFrame) -> pd.DataFrame:
        
        df_mRNA = df_mRNA.drop(columns=["Hugo_Symbol", "Entrez_Gene_Id"], axis=0)
        
        df_mRNA = df_mRNA.T.reset_index()
        
        df_mRNA = df_mRNA.rename(columns={"index":"Sample ID"})
        
        df_merged = pd.merge(df_mRNA, df_clinical_data, right_on="Sample ID", left_on="Sample ID")
        
        return df_merged
        
        
    
    def initialize_limma(self, df : pd.DataFrame, column : str) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        
        metadata = pd.DataFrame(df[column], index=df.index)
        
        metadata.columns = [column]
        
        number_data = df.drop(columns=[column])
        
        number_data = np.log2(number_data + 1)
        
        expr = number_data.T  #(Samples, Genes)
        
        assert (expr.columns == metadata.index).all() # type: ignore
        
        metadata_aligned = metadata.loc[expr.columns].copy() # type: ignore
        
        design = pd.get_dummies(metadata_aligned[column]).astype(float)
        
        #This is for fitting the models
        limma_fit_models = limma.lmFit(obj=expr, design=design)

        #Emperical moderate Bayes (eBayes)
        limma_fit_models = limma.eBayes(limma_fit_models)

        #Obtain the table of Results
        results = limma.topTable(limma_fit_models, number=np.inf) # type: ignore
        
        #Transform to pandas dataframe
        results_df = pd.DataFrame(results)
        if column == "Tumor-Cancer":
            results_df = results_df.rename(columns={
                                "column0":"HER2-enriched",
                                "column1":"Luminal A",
                                "column2":"Luminal B",
                                "column3":"TNBC"                                
                                })
            
        return (results_df, design, expr)
    
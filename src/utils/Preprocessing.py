import pandas as pd

def classify_cancer_type(df_clinical_data : pd.DataFrame) -> list:
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
            
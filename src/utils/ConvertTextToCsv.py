import pandas as pd

def TextToCsv(file_path : str) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep="\\t", engine="python")
    print(f"Shape of the CSV: {df.shape}")
    return df
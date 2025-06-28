import pandas as pd

def load_data(file_path):
    """Load CSV data and handle missing values."""
    try:
        df = pd.read_csv(file_path)
        df.fillna(method="ffill", inplace=True)
        print(f"[✔] Loaded data from: {file_path} — {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"[✘] Failed to load data: {e}")
        raise

import pandas as pd
import os

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load CSV data into a pandas DataFrame.
    
    Args:
        file_path (str) : Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File Not found: {file_path}")
    
    return pd.read_csv(file_path)
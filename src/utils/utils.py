import pandas as pd

def ensure_datetime(df, column):
    if not pd.api.types.is_datetime64_any_dtype(df[column]):
        df[column] = pd.to_datetime(df[column], errors='coerce')
    return df
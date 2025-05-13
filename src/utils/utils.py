import pandas as pd
from ast import literal_eval

def ensure_datetime(df, column):
    if not pd.api.types.is_datetime64_any_dtype(df[column]):
        df[column] = pd.to_datetime(df[column], errors='coerce')
    return df

def safe_parse_feat(x):
    try:
        if isinstance(x, str):
            parsed = literal_eval(x)
            if isinstance(parsed, list) and all(isinstance(i, (int, float)) for i in parsed):
                return parsed
            else:
                return None
        elif isinstance(x, list) and all(isinstance(i, (int, float)) for i in x):
            return x
        else:
            return None
    except:
        return None
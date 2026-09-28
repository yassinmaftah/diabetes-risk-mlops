import pandas as pd
import numpy as np




def drop_column(df:pd.DataFrame, column:str) -> pd.DataFrame :
    if column not in df.columns:
        raise KeyError(f"We don't have this Column [{column}] in our dataset")
    return df.drop(columns=[column])

def zeros_to_nan(df:pd.DataFrame, column:str) -> pd.DataFrame :
    df = df.copy()
    if column not in df.columns:
        raise KeyError(f"We don't have this Column [{column}] in our dataset")
    
    df[column] = df[column].replace(0,np.nan)
    return df

def choose_strategy(df: pd.DataFrame, column: str) -> str:
    if column not in df.columns:
        raise KeyError(f"We don't have this Column [{column}] in our dataset")

    skewness = df[column].skew()
    return "mean" if abs(skewness) < 0.5 else "median"

def impute_column(df: pd.DataFrame, column: str, strategy: str) -> pd.DataFrame:
    """Return a copy of the DataFrame where NaN in `column` are filled."""
    if column not in df.columns:
        raise KeyError(f"We don't have this Column [{column}] in our dataset")
    if strategy not in ("mean", "median"):
        raise ValueError(f"Error : '{strategy}' Use mean or median")

    df = df.copy()
    fill_value = df[column].mean() if strategy == "mean" else df[column].median()
    df[column] = df[column].fillna(fill_value)
    return df

def handle_missing(df: pd.DataFrame, column: str) -> pd.DataFrame:
    strategy = choose_strategy(df, column)
    return impute_column(df, column, strategy)
import pandas as pd
def drop_column(df:pd.DataFrame, column:str) -> pd.DataFrame :
    if column not in df.columns:
        raise KeyError(f"We don't have this Column [{column}] in our dataset")
    return df.drop(columns=[column])


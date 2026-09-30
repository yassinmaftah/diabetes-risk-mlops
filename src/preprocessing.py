import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from src.eda import graphe_check_outliers

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


def scale_column(df: pd.DataFrame, columns:list[str]) :
    for c in columns :
        if c not in df.columns:
            raise KeyError(f"We dno't have this column: [{c}]")
        
    df = df.copy()
    scaler = StandardScaler()
    
    df[columns] = scaler.fit_transform(df[columns])
    
    return df, scaler

def knn(df: pd.DataFrame, columns_z: list[str],others_column ,neighbors:int=5) :
    
    all_columns = columns_z + others_column
    df_scaled, scaler = scale_column(df,all_columns)
    
    imputer = KNNImputer(n_neighbors=neighbors)
    df1 = imputer.fit_transform(df_scaled[all_columns])
    df_final = scaler.inverse_transform(df1)
    
    df= df.copy()
    df[all_columns] = df_final
    return df



# Glucose OK 
# BloodPressure 3 value less then min 
# SkinThickness OK
# Insulin max = 300 and we have 37
# BMI OK
# DiabetesPedigreeFunction OK 
# Age OK 

def cap_column(df: pd.DataFrame, column: str, min_value: float = None, max_value: float = None) -> pd.DataFrame:
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found. Available: {list(df.columns)}")
    if min_value is None and max_value is None:
        raise ValueError("Provide at least one of min_value or max_value.")

    df = df.copy()
    df[column] = df[column].clip(lower=min_value, upper=max_value)
    return df
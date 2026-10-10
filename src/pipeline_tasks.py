from pathlib import Path

import pandas as pd

FEATURES = ["Glucose", "BloodPressure", "SkinThickness", "Insulin",
            "BMI", "DiabetesPedigreeFunction", "Age"]


def extract_data(path) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("The dataset is empty")
    print(f"Extracted {len(df)} rows")
    return str(path)


def preprocess_data(path, output_path) -> str:
    df = pd.read_csv(path).drop_duplicates()
    missing = df[FEATURES + ["Cluster"]].isna().sum().sum()
    if missing > 0:
        raise ValueError(f"{missing} missing values found")
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")
    return str(output_path)
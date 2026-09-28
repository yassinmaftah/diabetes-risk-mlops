import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def column_info(df: pd.DataFrame, column: str) -> pd.DataFrame:
        
    if column not in df.columns:
        raise KeyError(f"We don't have this Column [{column}] in our dataset")


    s = df[column]
    n = len(s)
    
    n_missing = s.isna().sum()
    n_zeros = (s == 0).sum()
    
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_outliers = ((s < low) | (s > high)).sum()
    
    rows = [
        ("General",      "Data type",        str(s.dtype)),
        ("General",      "Total rows",       n),
        ("General",      "Unique values",    s.nunique()),
        ("Quality",      "Missing (NaN)",    f"{n_missing} ({n_missing / n:.1%})"),
        ("Quality",      "Zeros",            f"{n_zeros} ({n_zeros / n:.1%})"),
        ("Distribution", "Min",              round(s.min(), 2)),
        ("Distribution", "Max",              round(s.max(), 2)),
        ("Distribution", "Mean",             round(s.mean(), 2)),
        ("Distribution", "Median",           round(s.median(), 2)),
        ("Distribution", "Std deviation",    round(s.std(), 2)),
        ("Distribution", "Skewness",         round(s.skew(), 2)),
        ("Outliers",     "Q1 (25%)",         round(q1, 2)),
        ("Outliers",     "Q3 (75%)",         round(q3, 2)),
        ("Outliers",     "Lower limit",      round(low, 2)),
        ("Outliers",     "Upper limit",      round(high, 2)),
        ("Outliers",     "Outliers count",   f"{n_outliers} ({n_outliers / n:.1%})"),
    ]

    return pd.DataFrame(rows, columns=["Section", "Metric", "Value"])


def graphe_check_outliers(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        raise KeyError(f"We don't have this Column [{column}] in our dataset")
    
    fig, (ax_box, ax_hist) = plt.subplots(1, 2, figsize=(12, 4))

    sns.boxplot(x=df[column], ax=ax_box, color="lightblue")
    ax_box.set_title("Boxplot (dots = outliers)")

    sns.histplot(df[column], kde=True, ax=ax_hist)
    ax_hist.set_title("Distribution")

    fig.suptitle(column)
    plt.tight_layout()
    plt.show()
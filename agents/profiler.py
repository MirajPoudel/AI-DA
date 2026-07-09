import pandas as pd
import numpy as np


def profile_dataset(df: pd.DataFrame) -> dict:
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    profile = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
    }

    if numeric_cols:
        profile["numeric_summary"] = df[numeric_cols].describe().to_dict()
        if len(numeric_cols) > 1:
            profile["correlations"] = df[numeric_cols].corr().round(2).to_dict()

    return profile

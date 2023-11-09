import pandas as pd
import numpy as np
from typing import Optional


def rotate_traces(
    df: pd.DataFrame, increment: Optional[int] = None, time_col: str = "time"
) -> pd.DataFrame:
    if time_col not in df.columns:
        raise ValueError(f"'{time_col}' not found in DataFrame's columns.")

    if increment is None:
        increment = np.random.randint(0, len(df) - 1)
    else:
        # Handle negative increments
        increment = increment % len(df)

    df = df.copy()

    # Separate time column and other columns
    time_data = df[time_col]
    other_data = df.drop(columns=[time_col])

    # Rotate the other columns
    rotated_vals = np.roll(other_data.values, increment, axis=0)

    # Create new dataframe with rotated values and original time column
    rotated_df = pd.DataFrame(rotated_vals, columns=other_data.columns)

    rotated_df[time_col] = time_data
    rotated_df = rotated_df[[time_col] + list(other_data.columns)]
    return rotated_df

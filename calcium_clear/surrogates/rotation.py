import pandas as pd
import numpy as np
from typing import Optional


def rotate_traces(
    df: pd.DataFrame,
    increment: Optional[int] = None,
    time_col: str = "time",
    copy: bool = True,
) -> pd.DataFrame:
    """
    Rotate the traces in a DataFrame.

    This function rotates the traces in the DataFrame `df` by a specified `increment`. The rotation is performed on the rows of the DataFrame, excluding the `time_col`. If `increment` is not provided, a random increment is chosen.
    Uses np.roll to perform the rotation.

    Args:
        df (pd.DataFrame): The DataFrame containing the traces to rotate.
        increment (int, optional): The number of positions to rotate the traces. If not provided, a random increment is chosen. Defaults to None.
        time_col (str, optional): The name of the time column. Defaults to "time".

    Returns:
        pd.DataFrame: The DataFrame with rotated traces.

    Raises:
        ValueError: If `time_col` is not found in DataFrame's columns.

    Examples:
        >>> df = pd.DataFrame({
        ...     "time": [1, 2, 3, 4, 5],
        ...     "trace1": [10, 20, 30, 40, 50],
        ...     "trace2": [60, 70, 80, 90, 100],
        ... })
        >>> rotated_df = rotate_traces(df, increment=1, time_col="time")
    """
    if time_col not in df.columns:
        raise ValueError(f"'{time_col}' not found in DataFrame's columns.")

    if increment is None:
        increment = np.random.randint(0, len(df) - 1)
    else:
        increment = increment % len(df)  # negative increments

    if copy:
        df = df.copy()

    time_data = df[time_col]
    other_data = df.drop(columns=[time_col])

    rotated_vals = np.roll(other_data.values, increment, axis=0)

    rotated_df = pd.DataFrame(rotated_vals, columns=other_data.columns)

    rotated_df[time_col] = time_data
    rotated_df = rotated_df[[time_col] + list(other_data.columns)]
    return rotated_df

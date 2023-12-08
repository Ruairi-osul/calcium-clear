import pandas as pd
import numpy as np


def sample_traces(
    df_wide: pd.DataFrame,
    time_col: str = "time",
    n_retained: int | None = None,
    frac_retained: float | None = None,
    with_replacement: bool = False,
    other_cols: list | None = None,
) -> pd.DataFrame:
    """
    Sample columns from a wide-format DataFrame, retaining specified columns.

    This function pops the time series and other specified columns from the input DataFrame,
    then samples the remaining columns. The retained and sampled columns are concatenated
    and returned.

    Args:
        df_wide (pd.DataFrame): The input DataFrame in wide format.
        time_col (str, optional): The name of the time column. Defaults to "time".
        n_retained (int | None, optional): The number of columns to retain. If None, all columns are retained. Defaults to None.
        frac_retained (float | None, optional): The fraction of columns to retain. If None, all columns are retained. Defaults to None.
        with_replacement (bool, optional): Whether to sample with replacement. Defaults to False.
        other_cols (list | None, optional): List of other column names to retain. If None, no other columns are retained. Defaults to None.

    Returns:
        pd.DataFrame: The output DataFrame with the retained and sampled columns.

    Examples:
        >>> sample_traces(df, "time", 2, 0.5, True, ["col1", "col2"])
    """
    time_ser = df_wide.pop(time_col)
    match other_cols:
        case None:
            retained_cols = [time_ser]
        case _:
            retained_cols = [time_ser] + [df_wide.pop(col) for col in other_cols]

    df_retained = pd.concat(retained_cols, axis="columns")

    df_sub = df_wide.sample(
        frac=frac_retained, n=n_retained, replace=with_replacement, axis="columns"
    )

    df_out = pd.concat([df_retained, df_sub], axis="columns")
    return df_out

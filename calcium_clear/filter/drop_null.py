from typing import Optional, Union
import pandas as pd


def drop_null_traces(
    df_wide: pd.DataFrame,
    thresh: Optional[int] = 5000,
) -> pd.DataFrame:
    """
    Drop traces that have less than a threshold number of non-null values.

    Args:
        df_wide (pd.DataFrame): wide dataframe with traces
        thresh (int, optional): threshold for number of non-null values. Defaults to 5000.

    Returns:
        pd.DataFrame: dataframe with null traces dropped
    """
    # Validate dataframe inputs
    assert isinstance(df_wide, pd.DataFrame), "df_wide should be a pandas DataFrame"

    # Validate thresh
    if thresh is not None:
        assert isinstance(thresh, int), "thresh should be an integer"
        assert thresh >= 0, "thresh should be >= 0"

    df_wide = df_wide.dropna(thresh=thresh, axis=1)
    return df_wide

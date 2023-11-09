from typing import Callable, Optional, Sequence, Union
import pandas as pd


def drop_null_traces(
    df_wide: pd.DataFrame,
    thresh: Optional[int] = 5000,
) -> pd.DataFrame:
    """
    Drop columns in a wide-format dataframe that don't have at least `thresh` non-NA values.

    Parameters
    ----------
    df_wide
        Wide-format dataframe.
    thresh
        Require that many non-NA values.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe.


    """
    # Validate dataframe inputs
    assert isinstance(df_wide, pd.DataFrame), "df_wide should be a pandas DataFrame"

    # Validate thresh
    if thresh is not None:
        assert isinstance(thresh, int), "thresh should be an integer"
        assert thresh >= 0, "thresh should be >= 0"

    df_wide = df_wide.dropna(thresh=thresh, axis=1)
    return df_wide

import pandas as pd
from typing import Sequence


def select_columns(
    df: pd.DataFrame,
    cols: Sequence[str],
    inclusive: bool = True,
    on_error: str = "raise",
) -> pd.DataFrame:
    """
    Select a pre-specified set of columns from a pandas DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to select columns from.
        cols (Sequence[str]): List of column names to select.
        inclusive (bool): Whether to select the columns specified in `cols`
            (True) or to select all columns except those specified in `cols` (False).
            Defaults to True.
        on_error (Optional[str]): How to handle the case where one or more columns
            specified in `cols` are not found in `df`. If 'raise', raise a ValueError.
            If 'skip', return only the columns that are found in `df`. If 'pass',
            return the original DataFrame. Defaults to 'raise'.


    Returns:
        pd.DataFrame: DataFrame with selected columns.
    """
    if not set(cols).issubset(df.columns):
        missing_cols = list(set(cols) - set(df.columns))
        if on_error == "raise":
            raise ValueError(f"Columns {missing_cols} not found in DataFrame.")
        elif on_error == "skip":
            cols = list(set(cols).intersection(set(df.columns)))
        elif on_error == "pass":
            return df
        else:
            raise ValueError(
                "on_error parameter must be one of ['raise', 'skip', 'pass']."
            )

    if inclusive:
        return df[cols]
    else:
        return df[df.columns.difference(cols)]

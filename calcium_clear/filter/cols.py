import pandas as pd
from typing import Sequence


def select_columns(
    df: pd.DataFrame,
    cols: Sequence[str],
    inclusive: bool = True,
    on_error: str = "raise",
):
    """
    Select a pre-specified set of columns from a pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    cols : list
        List of columns to select.
    inclusive : bool, optional
        If True, select specified columns,
        if False, exclude specified columns. Default is True.
    on_error : str, optional
        What to do if specified columns are not found. 'raise' to raise an error,
        'skip' to ignore missing columns, 'pass' to pass the entire DataFrame
        if any column is missing. Default is 'raise'.

    Returns
    -------
    pd.DataFrame
        Output DataFrame.
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

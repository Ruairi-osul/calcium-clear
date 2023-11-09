import pandas as pd
from typing import Optional, Sequence, Union


def _validate_inputs(
    df_wide: pd.DataFrame,
    df_meta: pd.DataFrame,
    df_wide_other_cols: Optional[Union[str, Sequence[str]]],
    df_meta_group_col: str,
    df_meta_cell_col: str,
):
    """
    Validate the inputs to the function filter_by_group.

    Parameters
    ----------
    df_wide
        Wide-format dataframe.
    df_meta
        Metadata dataframe.
    df_wide_other_cols
        Other columns to include in the filtered dataframe.
    df_meta_group_col
        Column in metadata dataframe that contains group information.
    df_meta_cell_col
        Column in metadata dataframe that contains cell information.

    Returns
    -------
    None
    """
    # Validate dataframe inputs
    assert isinstance(df_wide, pd.DataFrame), "df_wide should be a pandas DataFrame"
    assert isinstance(df_meta, pd.DataFrame), "df_meta should be a pandas DataFrame"

    # Validate column names in df_wide
    df_wide_cols = set(df_wide.columns)
    if isinstance(df_wide_other_cols, str):
        assert (
            df_wide_other_cols in df_wide_cols
        ), f"df_wide does not contain column {df_wide_other_cols}"
    elif isinstance(df_wide_other_cols, Sequence):
        assert set(df_wide_other_cols).issubset(
            df_wide_cols
        ), f"df_wide does not contain columns {df_wide_other_cols}"

    # Validate column names in df_meta
    df_meta_cols = set(df_meta.columns)
    assert (
        df_meta_group_col in df_meta_cols
    ), f"df_meta does not contain column {df_meta_group_col}"
    assert (
        df_meta_cell_col in df_meta_cols
    ), f"df_meta does not contain column {df_meta_cell_col}"

    # Validate that df_meta_cell_col in df_meta corresponds to columns in df_wide
    assert set(df_meta[df_meta_cell_col].unique()).issubset(
        df_wide_cols
    ), f"df_meta[{df_meta_cell_col}] does not correspond to columns in df_wide"


def filter_by_group(
    df_wide: pd.DataFrame,
    df_meta: pd.DataFrame,
    group: Union[str, Sequence[str]],
    df_wide_other_cols: Optional[Union[str, Sequence[str]]] = "time",
    df_meta_group_col: str = "group",
    df_meta_cell_col: str = "cell_id",
    inclusive: bool = True,
):
    """
    Filter a wide-format dataframe by group.

    Parameters
    ----------
    df_wide
        Wide-format dataframe.
    df_meta
        Metadata dataframe.
    group
        Group(s) to filter by.
    df_wide_other_cols
        Other columns to include in the filtered dataframe.
    df_meta_group_col
        Column in metadata dataframe that contains group information.
    df_meta_cell_col
        Column in metadata dataframe that contains cell information.
    inclusive
        Whether to include or exclude the group(s).

    Returns
    -------
    pd.DataFrame
        Filtered dataframe.

    Example
    -------
    >>> df_wide = pd.DataFrame(...)
    >>> df_meta = pd.DataFrame(...)
    >>> group = 'group1'
    >>> df_wide_other_cols = ['time']
    >>> df_meta_group_col = 'group'
    >>> df_meta_cell_col = 'cell_id'
    >>> inclusive = True
    >>> filtered_df = filter_by_group(df_wide, df_meta, group, df_wide_other_cols, df_meta_group_col, df_meta_cell_col, inclusive)
    """
    # Validate inputs
    _validate_inputs(
        df_wide, df_meta, df_wide_other_cols, df_meta_group_col, df_meta_cell_col
    )

    if isinstance(group, str):
        group = [group]

    if df_wide_other_cols is not None:
        if isinstance(df_wide_other_cols, str):
            df_wide_other_cols = [df_wide_other_cols]
        else:
            df_wide_other_cols = list(df_wide_other_cols)
    else:
        df_wide_other_cols = []

    if inclusive:
        df_meta = df_meta.loc[df_meta[df_meta_group_col].isin(group)]
    else:
        df_meta = df_meta.loc[~df_meta[df_meta_group_col].isin(group)]

    cols_to_keep = df_meta[df_meta_cell_col].unique().tolist() + df_wide_other_cols
    df_wide = df_wide.loc[:, cols_to_keep]

    return df_wide

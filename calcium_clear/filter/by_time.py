from typing import Optional, Callable
import pandas as pd


def _validate_inputs(
    df_wide: pd.DataFrame,
    time_col: str,
    t_start: Optional[float],
    t_stop: Optional[float],
    custom_filter: Optional[Callable[[pd.DataFrame], pd.Series]],
):
    """
    Validate the inputs to the function filter_by_time.

    Parameters
    ----------
    df_wide
        Wide-format dataframe.
    time_col
        Column in df_wide that contains time information.
    t_start
        Start time for filtering.
    t_stop
        Stop time for filtering.
    custom_filter
        Custom filter function that takes a DataFrame and returns a Series of booleans.

    Returns
    -------
    None
    """
    # Validate dataframe inputs
    assert isinstance(df_wide, pd.DataFrame), "df_wide should be a pandas DataFrame"

    # Validate time column
    assert (
        time_col in df_wide.columns
    ), f"'{time_col}' not found in DataFrame's columns."

    # Validate t_start and t_stop
    if t_start is not None and t_stop is not None:
        assert t_start <= t_stop, f"t_start ({t_start}) must be <= t_stop ({t_stop})."

    # Validate custom_filter
    if custom_filter is not None:
        assert callable(custom_filter), "custom_filter must be a function"


def filter_by_time(
    df_wide: pd.DataFrame,
    time_col: str = "time",
    t_start: Optional[float] = None,
    t_stop: Optional[float] = None,
    custom_filter: Optional[Callable[[pd.DataFrame], pd.Series]] = None,
) -> pd.DataFrame:
    """
    Filter a wide-format dataframe by time.

    Parameters
    ----------
    df_wide
        Wide-format dataframe.
    time_col
        Column in df_wide that contains time information.
    t_start
        Start time for filtering.
    t_stop
        Stop time for filtering.
    custom_filter
        Custom filter function that takes a DataFrame and returns a Series of booleans.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe.

    """
    # Validate inputs
    _validate_inputs(df_wide, time_col, t_start, t_stop, custom_filter)

    if t_start is not None:
        df_wide = df_wide.loc[df_wide[time_col] >= t_start]

    if t_stop is not None:
        df_wide = df_wide.loc[df_wide[time_col] <= t_stop]

    if custom_filter is not None:
        df_wide = df_wide.loc[custom_filter(df_wide)]

    return df_wide

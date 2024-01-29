from typing import Optional, Callable, TypeVar
import pandas as pd

T = TypeVar("T")


def _validate_inputs(
    df_wide: pd.DataFrame,
    time_col: str,
    t_start: Optional[float],
    t_stop: Optional[float],
    custom_filter: Optional[Callable[[T], T]],
):
    """
    Validate the inputs to the function filter_by_time.

    Args:
        ff_wide (pd.DataFrame): Wide-format dataframe.
        time_col (str): Column in df_wide that contains time information.
        t_start (Optional[float]): Start time for filtering.
        t_stop (Optional[float]): Stop time for filtering.
        custom_filter (Optional[Callable[[T], T]]): Custom filter function that takes a DataFrame and returns a Series of booleans.

    Returns:
    - None
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
    custom_filter: Optional[Callable[[T], T]] = None,
) -> pd.DataFrame:
    """
    Filter a wide-format dataframe by time.

    Args:
        df_wide (pd.DataFrame): Wide-format dataframe.
        time_col (str): Column in df_wide that contains time information.
        t_start (Optional[float]): Start time for filtering.
        t_stop (Optional[float]): Stop time for filtering.
        custom_filter (Optional[Callable[[T], T]]): Custom filter function that takes a DataFrame and returns a Series of booleans.

    Returns:
        pd.DataFrame: the filtered dataframe
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

import pandas as pd
from typing import Optional, List, Dict, Union


def resample_traces(
    df_wide: pd.DataFrame,
    time_col: str,
    resample_frequency: float = 0.1,
    resample_strategy: str = "ffill",
) -> pd.DataFrame:
    df_wide.loc[:, time_col] = pd.to_timedelta(df_wide[time_col], unit="s")
    df_wide.set_index(time_col, inplace=True)

    df_resampled = df_wide.resample(f"{resample_frequency}S").agg(resample_strategy)
    df_resampled.reset_index(inplace=True)
    df_resampled[time_col] = df_resampled[time_col].dt.total_seconds()
    return df_resampled


def resample_traces_specify(
    df_wide: pd.DataFrame,
    time_col: str,
    resample_frequency: float = 0.1,
    object_resample_strategy: str = "ffill",
    numeric_resample_strategy: str = "ffill",
    column_resample_strategy: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Resample a DataFrame with mixed data types, applying different strategies
    to object and numeric columns, and to a specified set of named columns.

    This function uses the pandas DataFrame.resample() method to downsample
    or upsample the dataset, with strategies provided by the user.

    Args:
        df_wide (pd.DataFrame): Input dataframe to be resampled.
        time_col (str): The name of the column in df_wide to be used as time index.
        resample_frequency (float, optional): Frequency of resampling in seconds.
        object_resample_strategy (str, optional): Strategy to apply to object columns during resampling.
            Default is forward fill ("ffill").
        numeric_resample_strategy (str, optional): Strategy to apply to numeric columns during resampling.
            Default is forward fill ("ffill").
        column_resample_strategy (dict, optional): Dictionary mapping column names to resampling strategies.
            If provided, these strategies will override the default behavior for those columns.

    Returns:
        pd.DataFrame: Resampled dataframe.

    Example:
        >>> df = pd.DataFrame({
        ...     "time": range(10),
        ...     "str_col": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        ...     "num_col": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10]
        ... })
        >>> df_resampled = resample_traces_v2(
        ...     df,
        ...     "time",
        ...     2,
        ...     object_resample_strategy="bfill",
        ...     numeric_resample_strategy="max",
        ...     column_resample_strategy={"str_col": "ffill"}
        ... )
        >>> print(df_resampled)
    """

    df_wide[time_col] = pd.to_timedelta(df_wide[time_col], unit="s")
    df_wide.set_index(time_col, inplace=True)

    resample_strategy_mapping = {}

    for column in df_wide.columns:
        if column_resample_strategy and column in column_resample_strategy:
            resample_strategy_mapping[column] = column_resample_strategy[column]
        elif df_wide[column].dtype == "object":
            resample_strategy_mapping[column] = object_resample_strategy
        else:
            resample_strategy_mapping[column] = numeric_resample_strategy

    df_resampled = df_wide.resample(f"{resample_frequency}S").agg(
        resample_strategy_mapping
    )

    df_resampled.reset_index(inplace=True)
    df_resampled[time_col] = df_resampled[time_col].dt.total_seconds()

    return df_resampled

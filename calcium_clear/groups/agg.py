from typing import Callable, Dict, List, Union
from .mapper import validate_mapper
import pandas as pd


def aggregate_groups_time(
    df_wide: pd.DataFrame,
    group_dict: Dict[str, List[str]],
    agg_func: Union[str, Callable] = "mean",
    time_col: str = "time",
    handle_missing: str = "raise",
) -> pd.DataFrame:
    """
    Aggregates columns in a DataFrame based on groups specified in a dictionary.
    It applies an aggregation function to each group of columns and returns a DataFrame with the aggregated values.

    Args:
        df_wide (pandas.DataFrame): Input DataFrame
        group_dict (dict): Dictionary specifying the groups. Keys are group names, values are lists of column names in each group.
        agg_func (Union[str, Callable]): Function to use for aggregating the data. Can be a callable or a string specifying a pandas function.
        time_col (str): Name of the time column in the DataFrame.
        handle_missing (str, optional): How to handle missing columns in the groups. Options are 'raise', 'skip_group', 'skip_column', and 'warn'.

    Returns:
        pandas.DataFrame: DataFrame with the aggregated data. Same index as the input DataFrame, one column for each group.
    """

    # Prepare an empty DataFrame to hold the aggregated data
    agg_df = pd.DataFrame(index=df_wide.index)

    # Validate the group_dict
    group_dict = validate_mapper(df_wide, group_dict, missing_handling=handle_missing)

    # Iterate over each group in the input dictionary
    for group_name, cols in group_dict.items():
        # Perform the aggregation and add the result to agg_df
        agg_df[group_name] = df_wide[cols].agg(agg_func, axis=1)

    agg_df[time_col] = df_wide[time_col]
    return agg_df

import pandas as pd
from typing import Callable


def _validate_input(
    df: pd.DataFrame,
    group_column: str,
    aggregation_function: Callable[[pd.DataFrame], pd.Series],
):
    assert isinstance(df, pd.DataFrame), "df must be a pandas DataFrame."
    assert isinstance(group_column, str), "group_column must be a string."
    assert (
        group_column in df.columns
    ), f"{group_column} is not a column in the dataframe."
    assert callable(
        aggregation_function
    ), "aggregation_function must be a callable function."


def group_aggregate_pivot(df: pd.DataFrame, group_column: str, aggregation_function):
    _validate_input(df, group_column, aggregation_function)
    grouped = df.groupby(group_column).agg(aggregation_function)
    pivot = grouped.reset_index().pivot(columns=group_column)

    return pivot


def group_apply_pivot(
    df: pd.DataFrame,
    group_column: str,
    aggregation_function: Callable[[pd.DataFrame], pd.Series],
) -> pd.DataFrame:
    _validate_input(df, group_column, aggregation_function)
    # First, group the dataframe by the specified column and apply the aggregation function
    grouped = df.groupby(group_column).apply(aggregation_function).unstack().unstack()
    return grouped

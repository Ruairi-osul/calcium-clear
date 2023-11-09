import pandas as pd
from typing import List, Dict


def validate_mapper(
    df: pd.DataFrame, mapper: Dict[str, List[str]], missing_handling: str = "error"
) -> Dict[str, List[str]]:
    """
    Validate and handle missing columns in the mapper.

    Args:
        df (pandas.DataFrame): The dataframe for which to check the mapper.
        mapper (dict): A dictionary mapping group names to column names.
        missing_handling (str, optional): How to handle missing columns in the dataframe.
            Options are 'error', 'warn', or 'skip'. Default is 'error'.

    Returns:
        dict: The valid and possibly modified mapper.

    Raises:
        ValueError: If `missing_handling` is 'error' and there are missing columns.
        Warning: If `missing_handling` is 'warn' and there are missing columns.
    """
    valid_mapper = {}
    for group, columns in mapper.items():
        missing_columns = [column for column in columns if column not in df.columns]
        if missing_columns:
            if missing_handling == "error":
                raise ValueError(f"Columns {missing_columns} not in dataframe.")
            elif missing_handling == "warn":
                print(
                    f"Warning: Columns {missing_columns} not in dataframe. Excluding these columns."
                )
                columns = [
                    column for column in columns if column not in missing_columns
                ]
            elif missing_handling == "skip":
                continue
        valid_mapper[group] = columns
    return valid_mapper


import pandas as pd
from typing import Dict, List, Any


def get_group_values(
    df: pd.DataFrame, group_col: str, neuron_col: str
) -> Dict[Any, List[Any]]:
    """
    Returns a dictionary with keys being the unique value of a specified "group_col"
    and values being all unique values in that group in a specified "neuron_col".

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe.
    group_col : str
        The column in df to group by.
    neuron_col : str
        The column in df from which to get unique values.

    Returns
    -------
    dict
        The output dictionary.

    Raises
    ------
    KeyError
        If group_col or neuron_col is not a column in df.
    TypeError
        If df is not a pandas DataFrame.
    """

    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, but it is a {type(df)}")
    if group_col not in df.columns:
        raise KeyError(f"{group_col} is not a column in df")
    if neuron_col not in df.columns:
        raise KeyError(f"{neuron_col} is not a column in df")

    # Create a dictionary where the keys are unique values in group_col
    # and the values are sets of unique values in neuron_col for each group.
    grouped_values = df.groupby(group_col)[neuron_col].unique().to_dict()

    # Convert numpy arrays to lists for better compatibility with JSON and other formats.
    for key in grouped_values:
        grouped_values[key] = grouped_values[key].tolist()

    return grouped_values

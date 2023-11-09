import pandas as pd
from typing import Optional, List


def _min_max_drop(df_wide: pd.DataFrame) -> pd.DataFrame:
    return df_wide.apply(
        lambda x: (x - x.dropna().min()) / (x.dropna().max() - x.dropna().min()), axis=0
    )


def _min_max_keep(df_wide: pd.DataFrame) -> pd.DataFrame:
    return df_wide.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)


def min_max(
    df_wide: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None,
    drop_na: bool = True,
):
    if exclude_cols is None:
        exclude_cols = []

    include_cols = set(df_wide.columns) - set(exclude_cols)

    if drop_na:
        method = _min_max_drop
    else:
        method = _min_max_keep

    df_wide.loc[:, include_cols] = method(df_wide.loc[:, include_cols])

    return df_wide

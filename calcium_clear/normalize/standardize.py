import pandas as pd
import scipy.stats
from typing import Optional, List


def _zscore_drop(
    df_wide: pd.DataFrame,
) -> pd.DataFrame:
    return df_wide.apply(lambda x: (x - x.dropna().mean()) / x.dropna().std(), axis=0)


def _zscore_keep(df_wide: pd.DataFrame) -> pd.DataFrame:
    return df_wide.apply(scipy.stats.zscore, axis=0)


def zscore(
    df_wide: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None,
    drop_na: bool = True,
):
    if exclude_cols is None:
        exclude_cols = []

    include_cols = set(df_wide.columns) - set(exclude_cols)

    if drop_na:
        method = _zscore_drop
    else:
        method = _zscore_keep

    df_wide.loc[:, include_cols] = method(df_wide.loc[:, include_cols])

    return df_wide

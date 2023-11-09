from typing import Optional, Union, Callable
import pandas as pd
import numpy as np
from calcium_clear.stats import auc


def pre_post(
    df_aligned: pd.DataFrame,
    aligned_time_col: str = "aligned_time",
    zero_time: float = 0,
    time_col: Optional[str] = "time",
    compare_func: Union[str, Callable] = "auc",
) -> pd.DataFrame:
    df_auc = (
        df_aligned.assign(
            pre_post=np.where(df_aligned["aligned_time"] < 0, "pre", "post")
        )
        .drop(["aligned_time", "time"], axis=1)
        .groupby(["pre_post", "event_idx"])
        .agg(auc)
        .unstack()
        .transpose()
        .reset_index()
        .rename(
            columns={
                "level_0": "cell_id",
            }
        )
    )

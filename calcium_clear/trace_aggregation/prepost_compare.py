import pandas as pd
from typing import Optional, Union, Callable
import numpy as np
from calcium_clear.stats import auc
from functools import partial
from typing import Any
from .prepost_independent import prepost_agg


def prepost_diff(
    df_aligned_long: pd.DataFrame,
    aligned_time_col: str = "aligned_time",
    event_idx_col: Optional[str] = "event_idx",
    neuron_col: str = "neuron",
    value_col: str = "value",
    prepost_agg_func: Union[str, Callable] = "auc",
    pre_indicator: str = "pre",
    post_indicator: str = "post",
    time_sep: float = 0,
    created_diff_col: str = "post_sub_pre",
    drop_pre_post_cols: bool = False,
) -> pd.DataFrame:
    by_event = prepost_agg(
        df_aligned_long=df_aligned_long,
        aligned_time_col=aligned_time_col,
        event_idx_col=event_idx_col,
        value_col=value_col,
        neuron_col=neuron_col,
        agg_func=prepost_agg_func,
        pre_indicator=pre_indicator,
        post_indicator=post_indicator,
        time_sep=time_sep,
    )
    if event_idx_col is not None:
        by_event = by_event.sort_values(by=[neuron_col, event_idx_col])
    else:
        by_event = by_event.sort_values(by=[neuron_col])
    by_event[created_diff_col] = by_event[post_indicator] - by_event[pre_indicator]
    if drop_pre_post_cols:
        by_event = by_event.drop(columns=[pre_indicator, post_indicator])
    return by_event

import pandas as pd
from typing import Optional, Union, Callable
import numpy as np
from calcium_clear.stats import auc
from functools import partial


def _prepost_agg_groupby(
    df_aligned_long: pd.DataFrame,
    aligned_time_col: str = "aligned_time",
    event_idx_col: Optional[str] = "event_idx",
    neuron_col: str = "neuron",
    value_col: str = "value",
    agg_func: Union[str, Callable] = "auc",
    created_pre_post_col: str = "pre_post",
    pre_indicator: str = "pre",
    post_indicator: str = "post",
    time_sep: float = 0,
) -> pd.DataFrame:
    if event_idx_col is None:
        event_idx_col = "event_idx"
        df_aligned_long[event_idx_col] = 0

    if agg_func == "auc":
        agg_func = partial(auc, to_1=True)
    return (
        df_aligned_long.assign(
            **{
                created_pre_post_col: lambda x: np.where(
                    x[aligned_time_col] < time_sep, pre_indicator, post_indicator
                ),
            }
        )
        .groupby([event_idx_col, neuron_col, created_pre_post_col])[value_col]
        .apply(agg_func)
    )


def prepost_agg_long(
    df_aligned_long: pd.DataFrame,
    aligned_time_col: str = "aligned_time",
    event_idx_col: Optional[str] = "event_idx",
    neuron_col: str = "neuron",
    value_col: str = "value",
    agg_func: Union[str, Callable] = "auc",
    created_pre_post_col: str = "pre_post",
    pre_indicator: str = "pre",
    post_indicator: str = "post",
    time_sep: float = 0,
) -> pd.DataFrame:
    grouped = _prepost_agg_groupby(
        df_aligned_long=df_aligned_long,
        aligned_time_col=aligned_time_col,
        event_idx_col=event_idx_col,
        neuron_col=neuron_col,
        value_col=value_col,
        agg_func=agg_func,
        created_pre_post_col=created_pre_post_col,
        pre_indicator=pre_indicator,
        post_indicator=post_indicator,
        time_sep=time_sep,
    )
    return grouped.reset_index()


def prepost_agg(
    df_aligned_long: pd.DataFrame,
    aligned_time_col: str = "aligned_time",
    event_idx_col: Optional[str] = "event_idx",
    neuron_col: str = "neuron",
    value_col: str = "value",
    agg_func: Union[str, Callable] = "auc",
    created_pre_post_col: str = "pre_post",
    pre_indicator: str = "pre",
    post_indicator: str = "post",
    time_sep: float = 0,
) -> pd.DataFrame:
    grouped = _prepost_agg_groupby(
        df_aligned_long=df_aligned_long,
        aligned_time_col=aligned_time_col,
        event_idx_col=event_idx_col,
        neuron_col=neuron_col,
        value_col=value_col,
        agg_func=agg_func,
        created_pre_post_col=created_pre_post_col,
        pre_indicator=pre_indicator,
        post_indicator=post_indicator,
        time_sep=time_sep,
    )
    return grouped.unstack().reset_index().rename_axis(None, axis=1)

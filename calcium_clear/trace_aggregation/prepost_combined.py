import pandas as pd
from typing import Optional, Union, Callable
import numpy as np
from calcium_clear.stats import auc
from functools import partial
from typing import Any


def auc_post_minus_pre(
    df: pd.DataFrame,
    prepost_col: str = "pre_post",
    pre_indicator: str = "pre",
    post_indicator: str = "post",
    value_col: str = "value",
    to_1: bool = True,
) -> float:
    # print(df.head())
    pre = df[df[prepost_col] == pre_indicator][value_col].values
    post = df[df[prepost_col] == post_indicator][value_col].values
    return auc(post, to_1=to_1) - auc(pre, to_1=to_1)


def _event_agg_groupby(
    df_aligned_long: pd.DataFrame,
    aligned_time_col: str = "aligned_time",
    event_idx_col: Optional[str] = "event_idx",
    neuron_col: str = "neuron",
    value_col: str = "value",
    agg_func: Union[str, Callable[[pd.DataFrame], Any]] = "auc_post_minus_pre",
    created_pre_post_col: str = "pre_post",
    pre_indicator: str = "pre",
    post_indicator: str = "post",
    time_sep: float = 0,
) -> pd.DataFrame:
    if agg_func == "auc_post_minus_pre":
        agg_func = partial(
            auc_post_minus_pre,
            prepost_col=created_pre_post_col,
            pre_indicator=pre_indicator,
            post_indicator=post_indicator,
            value_col=value_col,
        )
    elif not callable(agg_func):
        raise ValueError(
            f"agg_func must be callable or 'auc_post_minus_pre', not {agg_func}"
        )

    if event_idx_col is None:
        event_idx_col = "event_idx"
        df_aligned_long[event_idx_col] = 0
    return (
        df_aligned_long.assign(
            **{
                created_pre_post_col: lambda x: np.where(
                    x[aligned_time_col] < time_sep, pre_indicator, post_indicator
                )
            }
        )
        .groupby([event_idx_col, neuron_col])
        .apply(agg_func)
    )


def event_agg_long(
    df_aligned_long: pd.DataFrame,
    aligned_time_col: str = "aligned_time",
    event_idx_col: str = "event_idx",
    neuron_col: str = "neuron",
    value_col: str = "value",
    agg_func: Union[str, Callable[[pd.DataFrame], Any]] = "auc_post_minus_pre",
    created_pre_post_col: str = "pre_post",
    pre_indicator: str = "pre",
    post_indicator: str = "post",
    time_sep: float = 0,
) -> pd.DataFrame:
    grouped = _event_agg_groupby(
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
    aggregated = grouped.reset_index()
    return aggregated


def event_agg(
    df_aligned_long: pd.DataFrame,
    aligned_time_col: str = "aligned_time",
    event_idx_col: str = "event_idx",
    neuron_col: str = "neuron",
    value_col: str = "value",
    agg_func: Union[str, Callable[[pd.DataFrame], Any]] = "auc_post_minus_pre",
    created_pre_post_col: str = "pre_post",
    pre_indicator: str = "pre",
    post_indicator: str = "post",
    time_sep: float = 0,
):
    grouped = _event_agg_groupby(
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

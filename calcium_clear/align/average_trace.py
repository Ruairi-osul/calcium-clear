from .align_events import align_to_events, align_to_events_grouped
import numpy as np
import pandas as pd
from typing import Union, Callable
from joblib import Parallel, delayed


def average_trace(
    df_wide: pd.DataFrame,
    events: np.ndarray,
    t_before: float,
    t_after: float,
    time_col: str = "time",
    created_aligned_time_col: str = "aligned_time",
    round_precision: int = 1,
    agg_func: Union[str, Callable] = "mean",
):
    df_aligned = align_to_events(
        df_wide,
        events=events,
        t_before=t_before,
        t_after=t_after,
        time_col=time_col,
        created_event_index_col="event_idx",
        round_precision=round_precision,
        created_aligned_time_col=created_aligned_time_col,
        drop_non_aligned=True,
    )
    df_average_trace = (
        df_aligned.drop(["event_idx", time_col], axis=1)
        .groupby(created_aligned_time_col)
        .agg(agg_func)
    )
    df_average_trace = df_average_trace.reset_index()
    return df_average_trace


def average_trace_long(
    df_wide: pd.DataFrame,
    events: np.ndarray,
    t_before: float,
    t_after: float,
    time_col: str = "time",
    created_aligned_time_col: str = "aligned_time",
    created_neuron_col: str = "neuron",
    created_value_col: str = "value",
    round_precision: int = 1,
    agg_func: Union[str, Callable] = "mean",
):
    df_average_trace = average_trace(
        df_wide=df_wide,
        events=events,
        t_before=t_before,
        t_after=t_after,
        time_col=time_col,
        created_aligned_time_col=created_aligned_time_col,
        round_precision=round_precision,
        agg_func=agg_func,
    )
    df_long = df_average_trace.melt(
        id_vars=[created_aligned_time_col],
        var_name=created_neuron_col,
        value_name=created_value_col,
    )

    return df_long


def average_trace_grouped(
    df_wide: pd.DataFrame,
    df_events: pd.DataFrame,
    t_before: float,
    t_after: float,
    df_wide_group_mapper: dict,
    round_precision: int = 1,
    n_jobs: int = -1,
    df_wide_time_col: str = "time",
    df_events_event_time_col: str = "event_time",
    df_events_group_col: str = "group",
    created_aligned_time_col: str = "aligned_time",
    agg_func: Union[str, Callable] = "mean",
):
    df_aligned = align_to_events_grouped(
        df_wide=df_wide,
        df_events=df_events,
        t_before=t_before,
        t_after=t_after,
        df_wide_group_mapper=df_wide_group_mapper,
        round_precision=round_precision,
        n_jobs=n_jobs,
        df_wide_time_col=df_wide_time_col,
        df_events_event_time_col=df_events_event_time_col,
        df_events_group_col=df_events_group_col,
        created_aligned_time_col=created_aligned_time_col,
    )
    df_average_trace = (
        df_aligned.drop(["event_idx"], axis=1)
        .groupby(created_aligned_time_col)
        .agg(agg_func)
    )
    df_average_trace = df_average_trace.reset_index()
    return df_average_trace


def average_trace_grouped_long(
    df_wide: pd.DataFrame,
    df_events: pd.DataFrame,
    t_before: float,
    t_after: float,
    df_wide_group_mapper: dict,
    round_precision: int = 1,
    n_jobs: int = -1,
    created_neuron_col: str = "neuron",
    created_value_col: str = "value",
    df_wide_time_col: str = "time",
    df_events_event_time_col: str = "event_time",
    df_events_group_col: str = "group",
    created_aligned_time_col: str = "aligned_time",
    agg_func: Union[str, Callable] = "mean",
):
    df_average_trace = average_trace_grouped(
        df_wide=df_wide,
        df_events=df_events,
        t_before=t_before,
        t_after=t_after,
        df_wide_group_mapper=df_wide_group_mapper,
        round_precision=round_precision,
        n_jobs=n_jobs,
        df_wide_time_col=df_wide_time_col,
        df_events_event_time_col=df_events_event_time_col,
        df_events_group_col=df_events_group_col,
        created_aligned_time_col=created_aligned_time_col,
        agg_func=agg_func,
    )
    df_long = df_average_trace.melt(
        id_vars=[created_aligned_time_col],
        var_name=created_neuron_col,
        value_name=created_value_col,
    )
    return df_long

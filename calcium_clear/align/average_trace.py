from .align_events import align_to_events, align_to_events_grouped
import numpy as np
import pandas as pd
from typing import Union, Callable


def average_trace(
    df_wide: pd.DataFrame,
    events: np.ndarray,
    t_before: float,
    t_after: float,
    time_col: str = "time",
    created_aligned_time_col: str = "aligned_time",
    round_precision: int = 1,
    agg_func: Union[str, Callable] = "mean",
) -> pd.DataFrame:
    """
    Aligns traces to events and averages them.

    Args:
        df_wide (pd.DataFrame): wide dataframe with traces
        events (np.ndarray): array of events to align to
        t_before (float): time before event to align to
        t_after (float): time after event to align to
        time_col (str, optional): name of time column. Defaults to "time".
        created_aligned_time_col (str, optional): name of aligned time column. Defaults to "aligned_time".
        round_precision (int, optional): precision to round to. Defaults to 1.
        agg_func (Union[str, Callable], optional): aggregation function to use. Defaults to "mean".

    Returns:
        pd.DataFrame: dataframe with average trace
    """
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
) -> pd.DataFrame:
    """
    Aligns traces to events and averages them, returning a long-format dataframe.

    Args:
        df_wide (pd.DataFrame): wide dataframe with traces
        events (np.ndarray): array of events to align to
        t_before (float): time before event to align to
        t_after (float): time after event to align to
        time_col (str, optional): name of time column. Defaults to "time".
        created_aligned_time_col (str, optional): name of aligned time column. Defaults to "aligned_time".
        created_neuron_col (str, optional): name of neuron column. Defaults to "neuron".
        created_value_col (str, optional): name of value column. Defaults to "value".
        round_precision (int, optional): precision to round to. Defaults to 1.
        agg_func (Union[str, Callable], optional): aggregation function to use. Defaults to "mean".

    Returns:
        pd.DataFrame: dataframe with average trace in long format.
    """
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
) -> pd.DataFrame:
    """
    Aligns and averages a dataframe of traces to a dataframe of events with each dataframe containing multiple groups.

    Average traces are formed from the traces in each group of the wide dataframe, aligned to the corresponding events in the events dataframe.

    Args:
        df_wide (pd.DataFrame): wide dataframe with traces
        df_events (pd.DataFrame): dataframe with events
        t_before (float): time before event to align to
        t_after (float): time after event to align to
        df_wide_group_mapper (dict): dictionary mapping groups in the wide dataframe to groups in the events dataframe
        round_precision (int, optional): precision to round to. Defaults to 1.
        n_jobs (int, optional): number of jobs to use for parallelization. Defaults to -1.
        df_wide_time_col (str, optional): name of time column in wide dataframe. Defaults to "time".
        df_events_event_time_col (str, optional): name of event time column in events dataframe. Defaults to "event_time".
        df_events_group_col (str, optional): name of group column in events dataframe. Defaults to "group".
        created_aligned_time_col (str, optional): name of aligned time column. Defaults to "aligned_time".
        agg_func (Union[str, Callable], optional): aggregation function to use. Defaults to "mean".

    Returns:
        pd.DataFrame: dataframe with average trace (wide format)
    """
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
        drop_time_col=True,
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
) -> pd.DataFrame:
    """
    ligns and averages a dataframe of traces to a dataframe of events with each dataframe containing multiple groups. Returns a long-format dataframe.

    Average traces are formed from the traces in each group of the wide dataframe, aligned to the corresponding events in the events dataframe.

    Args:
        df_wide (pd.DataFrame): wide dataframe with traces
        df_events (pd.DataFrame): dataframe with events
        t_before (float): time before event to align to
        t_after (float): time after event to align to
        df_wide_group_mapper (dict): dictionary mapping groups in the wide dataframe to groups in the events dataframe
        round_precision (int, optional): precision to round to. Defaults to 1.
        n_jobs (int, optional): number of jobs to use for parallelization. Defaults to -1.
        created_neuron_col (str, optional): name of neuron column. Defaults to "neuron".
        created_value_col (str, optional): name of value column. Defaults to "value".
        df_wide_time_col (str, optional): name of time column in wide dataframe. Defaults to "time".
        df_events_event_time_col (str, optional): name of event time column in events dataframe. Defaults to "event_time".
        df_events_group_col (str, optional): name of group column in events dataframe. Defaults to "group".
        created_aligned_time_col (str, optional): name of aligned time column. Defaults to "aligned_time".
        agg_func (Union[str, Callable], optional): aggregation function to use. Defaults to "mean".

    Returns:
        pd.DataFrame: dataframe with average trace (long format)
    """
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

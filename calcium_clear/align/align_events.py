import numpy as np
import pandas as pd
from binit import align_around, which_bin_idx
from typing import Optional
from joblib import Parallel, delayed


def align_to_events(
    df_wide: pd.DataFrame,
    events: np.ndarray,
    t_before: float,
    t_after: float,
    time_col: str = "time",
    created_event_index_col: str = "event_idx",
    created_aligned_time_col: str = "aligned_time",
    round_precision: int = 1,
    drop_non_aligned: bool = True,
):
    """
    Aligns a dataframe to events, creating a new column with the aligned time
    and a new column with the index of the event that was aligned to.

    Args:
        df_wide: A dataframe with a time column.
        events: A numpy array of event times.
        t_before: The time before the event to align to.
        t_after: The time after the event to align to.
        time_col: The name of the time column in df_wide.
        created_event_index_col: The name of the new column with the index of the
            event that was aligned to.
        round_precision: The number of decimal places to round the aligned time to.
        created_aligned_time_col: The name of the new column with the aligned time.
        drop_non_aligned: Whether to drop rows that were not aligned to an event.
    """
    events = np.asarray(events)
    df_wide[created_aligned_time_col] = align_around(
        df_wide[time_col].values, events, t_before=t_before, max_latency=t_after
    )
    df_wide[created_aligned_time_col] = df_wide[created_aligned_time_col].round(
        round_precision
    )

    df_wide[created_event_index_col] = which_bin_idx(
        df_wide[time_col].values, events, time_before=t_before, time_after=t_after
    )

    if drop_non_aligned:
        df_wide = df_wide.loc[df_wide[created_aligned_time_col].notnull()].copy()

    return df_wide


def align_to_events_long(
    df_wide: pd.DataFrame,
    events: np.ndarray,
    t_before: float,
    t_after: float,
    time_col: str = "time",
    created_event_index_col: str = "event_idx",
    created_aligned_time_col: str = "aligned_time",
    created_neuron_col: str = "neuron",
    created_value_col: str = "value",
    round_precision: int = 1,
    drop_non_aligned: bool = True,
):
    df_aligned = align_to_events(
        df_wide=df_wide,
        events=events,
        t_before=t_before,
        t_after=t_after,
        time_col=time_col,
        created_event_index_col=created_event_index_col,
        created_aligned_time_col=created_aligned_time_col,
        round_precision=round_precision,
        drop_non_aligned=drop_non_aligned,
    )
    df_long = df_aligned.melt(
        id_vars=[
            time_col,
            created_aligned_time_col,
            created_event_index_col,
        ],
        var_name=created_neuron_col,
        value_name=created_value_col,
    )
    return df_long


def align_to_events_grouped_long(
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
    created_event_index_col: str = "event_idx",
    created_aligned_time_col: str = "aligned_time",
    created_neuron_col: str = "neuron",
    created_value_col: str = "value",
    drop_non_aligned: bool = True,
):
    """

    Args:
        df_wide: A dataframe with a time column and a group column.
        df_events: A dataframe of event times and their respective group.
        t_before: The time before the event to align to.
        t_after: The time after the event to align to.
        df_wide_time_col: The name of the time column in df_wide.
        df_events_event_time_col: The name of the event time column in df_events.
        df_events_group_col: The name of the group column in df_events.
        created_event_index_col: The name of the new column in df_wide with the index
            of the event that was aligned to.
        created_aligned_time_col: The name of the new column in df_wide with the aligned time.
        drop_non_aligned: Whether to drop rows in df_wide that were not aligned to an event.

    Returns:
        df_wide: The input df_wide dataframe after alignment to the events. Each row now
        includes the aligned time and the index of the event that it was aligned to.

    """

    unique_groups_df = list(df_wide_group_mapper.keys())
    unique_groups_events = df_events[df_events_group_col].unique()

    def process_group(group):
        if group not in unique_groups_events:
            return None

        potential_group_cols = list(df_wide_group_mapper[group]) + [df_wide_time_col]

        df_group = df_wide[
            [c for c in potential_group_cols if c in df_wide.columns]
        ].copy()

        df_events_group = df_events.loc[df_events[df_events_group_col] == group].copy()

        df_group = align_to_events(
            df_group,
            df_events_group[df_events_event_time_col].values,
            t_before=t_before,
            t_after=t_after,
            round_precision=round_precision,
            time_col=df_wide_time_col,
            created_event_index_col=created_event_index_col,
            created_aligned_time_col=created_aligned_time_col,
            drop_non_aligned=drop_non_aligned,
        )
        df_group = df_group.melt(
            id_vars=[
                df_wide_time_col,
                created_aligned_time_col,
                created_event_index_col,
            ],
            var_name=created_neuron_col,
            value_name=created_value_col,
        )
        df_group = df_group.assign(**{df_events_group_col: group})
        return df_group

    df_list = Parallel(n_jobs=n_jobs)(
        delayed(process_group)(group) for group in unique_groups_df
    )
    df_list = [df for df in df_list if df is not None]

    df_long = pd.concat(df_list)
    return df_long


def align_to_events_grouped(
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
    created_event_index_col: str = "event_idx",
    created_aligned_time_col: str = "aligned_time",
    created_neuron_col: str = "neuron",
    created_value_col: str = "value",
    drop_non_aligned: bool = True,
):
    df_long = align_to_events_grouped_long(
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
        created_event_index_col=created_event_index_col,
        created_aligned_time_col=created_aligned_time_col,
        created_neuron_col=created_neuron_col,
        created_value_col=created_value_col,
        drop_non_aligned=drop_non_aligned,
    )
    return (
        df_long.pivot_table(
            index=[created_aligned_time_col, created_event_index_col],
            values=created_value_col,
            columns=created_neuron_col,
        )
        .sort_values(by=[created_aligned_time_col, created_event_index_col])
        .reset_index()
        .rename_axis(None, axis=1)
    )

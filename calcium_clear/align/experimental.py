from .align_events import align_to_events
from joblib import Parallel, delayed
import pandas as pd
import numpy as np


class EventAligner:
    def __init__(
        self,
        t_before: float,
        t_after: float,
        df_wide_time_col: str = "time",
        df_wide_group_col: str = "group",
        df_events_event_time_col: str = "event_time",
        df_events_group_col: str = "group",
        df_wide_created_event_index_col: str = "event_idx",
        created_aligned_time_col: str = "aligned_time",
        drop_non_aligned: bool = True,
        n_jobs: int = -1,
    ):
        self.t_before = t_before
        self.t_after = t_after
        self.df_wide_time_col = df_wide_time_col
        self.df_wide_group_col = df_wide_group_col
        self.df_events_event_time_col = df_events_event_time_col
        self.df_events_group_col = df_events_group_col
        self.df_wide_created_event_index_col = df_wide_created_event_index_col
        self.created_aligned_time_col = created_aligned_time_col
        self.drop_non_aligned = drop_non_aligned
        self.n_jobs = n_jobs

    def align_to_events(
        self,
        df_wide: pd.DataFrame,
        events: np.ndarray,
    ) -> pd.DataFrame:
        return align_to_events(
            df_wide=df_wide, events=events, t_before=self.t_before, t_after=self.t_after
        )

    def align_to_events_single_group(
        self, group: str, df_wide: pd.DataFrame, df_events: pd.DataFrame
    ) -> pd.DataFrame:
        df_group = df_wide.loc[df_wide[self.df_wide_group_col] == group].copy()
        df_events_group = df_events.loc[
            df_events[self.df_events_group_col] == group
        ].copy()

        df_group = self.align_to_events(
            df_group,
            df_events_group[self.df_events_event_time_col].values,
        )
        return df_group

    def align_to_events_grouped_parallel(
        self, df_wide: pd.DataFrame, df_events: pd.DataFrame
    ) -> pd.DataFrame:
        unique_groups = df_wide[self.df_wide_group_col].unique()

        df_list = Parallel(n_jobs=self.n_jobs)(
            delayed(self.align_to_events_single_group)(
                group,
                df_wide,
                df_events,
            )
            for group in unique_groups
        )

        df_wide = pd.concat(df_list, axis=0)
        return df_wide

"""Sleep stage metric parser."""

from __future__ import annotations

import pandas as pd

from samsung_health_sdk.metrics.base import BaseMetric
from samsung_health_sdk.utils import DateLike

SLEEP_STAGE_MAP: dict[int, str] = {
    40001: "Awake",
    40002: "Light",
    40003: "Deep",
    40004: "REM",
}


class SleepStageMetric(BaseMetric):
    """
    Parses com.samsung.health.sleep_stage.

    Each row is one contiguous sleep stage segment:
      start_time, end_time, stage (int code), stage_label (human-readable).

    The sleep_id column groups segments belonging to the same sleep session.
    """

    metric_name = "com.samsung.health.sleep_stage"
    value_columns = ["start_time", "end_time", "stage", "sleep_id",
                     "datauuid", "deviceuuid", "time_offset"]

    def load_summary(self, start: DateLike = None, end: DateLike = None) -> pd.DataFrame:
        df = super().load_summary(start, end)
        if "stage" in df.columns:
            df["stage"] = pd.to_numeric(df["stage"], errors="coerce")
            df["stage_label"] = df["stage"].map(SLEEP_STAGE_MAP)
        return df


class SleepSessionMetric(BaseMetric):
    """
    Parses com.samsung.shealth.sleep (session-level summaries).

    Provides total_duration, efficiency, and linked sleep stage data.
    """

    metric_name = "com.samsung.shealth.sleep"
    value_columns = []  # keep all columns — schema is complex


class SleepRawDataMetric(BaseMetric):
    """Parses com.samsung.shealth.sleep_raw_data (raw accelerometer/PPG based sleep)."""

    metric_name = "com.samsung.shealth.sleep_raw_data"
    value_columns = []

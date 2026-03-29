"""Step count metric parser."""

from __future__ import annotations

import pandas as pd

from samsung_health_sdk.metrics.base import BaseMetric
from samsung_health_sdk.utils import DateLike


class StepsMetric(BaseMetric):
    """
    Parses com.samsung.shealth.tracker.pedometer_step_count.

    Each row is a short (~1 min) step-count segment with walk_step, run_step,
    distance (metres), calorie, and speed.
    """

    metric_name = "com.samsung.shealth.tracker.pedometer_step_count"
    value_columns = [
        "start_time",
        "end_time",
        "count",
        "walk_step",
        "run_step",
        "distance",
        "calorie",
        "speed",
        "datauuid",
        "deviceuuid",
        "time_offset",
    ]

    def load_summary(self, start: DateLike = None, end: DateLike = None) -> pd.DataFrame:
        df = super().load_summary(start, end)
        for col in ("count", "walk_step", "run_step", "distance", "calorie", "speed"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df


class PedometerDaySummaryMetric(BaseMetric):
    """
    Parses com.samsung.shealth.tracker.pedometer_day_summary.

    Daily totals: step_count, distance, calorie, run_distance.
    """

    metric_name = "com.samsung.shealth.tracker.pedometer_day_summary"
    value_columns = []

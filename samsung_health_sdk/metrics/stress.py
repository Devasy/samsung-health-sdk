"""Stress metric parser."""

from __future__ import annotations

import pandas as pd

from samsung_health_sdk.metrics.base import BaseMetric
from samsung_health_sdk.utils import DateLike


class StressMetric(BaseMetric):
    """
    Parses com.samsung.shealth.stress.

    Summary columns: score (0-100), min, max per hourly window.
    Detail (binning): per-minute stress scores.
    """

    metric_name = "com.samsung.shealth.stress"
    value_columns = [
        "start_time",
        "end_time",
        "score",
        "min",
        "max",
        "datauuid",
        "deviceuuid",
        "time_offset",
    ]

    def load_summary(self, start: DateLike = None, end: DateLike = None) -> pd.DataFrame:
        df = super().load_summary(start, end)
        for col in ("score", "min", "max"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        rename = {}
        if "min" in df.columns:
            rename["min"] = "stress_min"
        if "max" in df.columns:
            rename["max"] = "stress_max"
        if rename:
            df = df.rename(columns=rename)
        return df

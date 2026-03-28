"""Heart rate metric parser."""

from __future__ import annotations


from samsung_health_sdk.metrics.base import BaseMetric
from samsung_health_sdk.utils import DateLike
import pandas as pd


class HeartRateMetric(BaseMetric):
    """
    Parses com.samsung.shealth.tracker.heart_rate.

    Summary columns: heart_rate (avg), min, max per session.
    Detail columns:  heart_rate, heart_rate_min, heart_rate_max per minute.
    """

    metric_name = "com.samsung.shealth.tracker.heart_rate"
    value_columns = [
        "start_time",
        "end_time",
        "heart_rate",
        "min",
        "max",
        "datauuid",
        "deviceuuid",
        "time_offset",
    ]

    def load_summary(self, start: DateLike = None, end: DateLike = None) -> pd.DataFrame:
        df = super().load_summary(start, end)
        # Rename min/max to heart_rate_min/max for clarity
        rename = {}
        if "min" in df.columns:
            rename["min"] = "heart_rate_min"
        if "max" in df.columns:
            rename["max"] = "heart_rate_max"
        if rename:
            df = df.rename(columns=rename)
        return df

    def load_detail(self, start: DateLike = None, end: DateLike = None) -> pd.DataFrame:
        df = super().load_detail(start, end)
        # Binning JSON uses heart_rate, heart_rate_min, heart_rate_max directly
        return df

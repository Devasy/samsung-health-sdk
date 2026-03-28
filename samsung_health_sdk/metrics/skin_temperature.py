"""Skin temperature metric parser."""

from __future__ import annotations

from samsung_health_sdk.metrics.base import BaseMetric
from samsung_health_sdk.utils import DateLike


class SkinTemperatureMetric(BaseMetric):
    """
    Parses com.samsung.health.skin_temperature.

    Summary columns: temperature (mean), min, max, baseline, stat_n (sample count).
    Detail (binning): per-minute temperature readings.
    """

    metric_name = "com.samsung.health.skin_temperature"
    value_columns = [
        "start_time", "end_time",
        "temperature", "min", "max", "baseline",
        "stat_n", "stat_m1", "stat_m2",
        "lower_bound", "upper_bound",
        "datauuid", "deviceuuid", "time_offset",
    ]

    def load_summary(self, start: DateLike = None, end: DateLike = None):
        df = super().load_summary(start, end)
        import pandas as pd
        for col in ("temperature", "min", "max", "baseline", "stat_n"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # Rename for clarity
        rename = {}
        if "min" in df.columns:
            rename["min"] = "temperature_min"
        if "max" in df.columns:
            rename["max"] = "temperature_max"
        if rename:
            df = df.rename(columns=rename)
        return df

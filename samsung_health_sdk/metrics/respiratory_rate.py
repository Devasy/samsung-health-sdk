"""Respiratory rate metric parser."""

from __future__ import annotations

import pandas as pd

from samsung_health_sdk.metrics.base import BaseMetric
from samsung_health_sdk.utils import DateLike


class RespiratoryRateMetric(BaseMetric):
    """
    Parses com.samsung.health.respiratory_rate.

    Summary: average breaths/min, upper_limit, lower_limit per session.
    Detail:  per-minute respiratory_rate from binning JSON files.
    """

    metric_name = "com.samsung.health.respiratory_rate"
    value_columns = [
        "start_time",
        "end_time",
        "average",
        "upper_limit",
        "lower_limit",
        "datauuid",
        "deviceuuid",
        "time_offset",
    ]

    def load_summary(self, start: DateLike = None, end: DateLike = None) -> pd.DataFrame:
        df = super().load_summary(start, end)
        for col in ("average", "upper_limit", "lower_limit"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

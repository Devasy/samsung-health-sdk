"""SpO2 (blood oxygen saturation) metric parser."""

from __future__ import annotations

import pandas as pd

from samsung_health_sdk.metrics.base import BaseMetric
from samsung_health_sdk.utils import DateLike


class SpO2Metric(BaseMetric):
    """
    Parses com.samsung.shealth.tracker.oxygen_saturation.

    Columns: spo2 (%), heart_rate (bpm at time of measurement).
    """

    metric_name = "com.samsung.shealth.tracker.oxygen_saturation"
    value_columns = [
        "start_time",
        "end_time",
        "spo2",
        "heart_rate",
        "datauuid",
        "deviceuuid",
        "time_offset",
    ]

    def load_summary(self, start: DateLike = None, end: DateLike = None) -> pd.DataFrame:
        df = super().load_summary(start, end)
        # Column may be named 'spo2' or 'oxygen_saturation' depending on export version
        if "oxygen_saturation" in df.columns and "spo2" not in df.columns:
            df = df.rename(columns={"oxygen_saturation": "spo2"})
        for col in ("spo2", "heart_rate"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

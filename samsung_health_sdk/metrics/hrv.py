"""Heart Rate Variability (HRV) metric parser."""

from __future__ import annotations

from samsung_health_sdk.metrics.base import BaseMetric
from samsung_health_sdk.utils import DateLike


class HRVMetric(BaseMetric):
    """
    Parses com.samsung.health.hrv.

    The CSV contains session-level metadata; detailed HRV measurements
    (SDNN, RMSSD, etc.) are stored in binning JSON files.
    """

    metric_name = "com.samsung.health.hrv"
    value_columns = ["start_time", "end_time", "datauuid", "deviceuuid", "time_offset"]

    def load_detail(self, start: DateLike = None, end: DateLike = None):
        """Return per-session HRV measurements from binning JSON files."""
        return super().load_detail(start, end)

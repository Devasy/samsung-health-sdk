"""Movement intensity metric parser."""

from __future__ import annotations

from samsung_health_sdk.metrics.base import BaseMetric
from samsung_health_sdk.utils import DateLike


class MovementMetric(BaseMetric):
    """
    Parses com.samsung.health.movement.

    Each row in the CSV covers a ~30–60 minute window.
    Binning JSON files contain per-minute 'activity_level' values —
    an accelerometer-derived movement intensity (higher = more movement).

    Typical ranges:
        0–5    sedentary (sitting, lying still)
        5–20   light (slow walking, fidgeting)
        20–50  low-moderate (walking)
        50–100 moderate (brisk walking)
        100+   vigorous (fast walking, running)
    """

    metric_name = "com.samsung.health.movement"
    value_columns = []  # keep all columns; activity_level comes from binning JSON

    def load_summary(self, start: DateLike = None, end: DateLike = None):
        """CSV-level summary rows (one per ~30–60 min session)."""
        return super().load_summary(start, end)

    def load_detail(self, start: DateLike = None, end: DateLike = None):
        """Per-minute activity_level from binning JSON files."""
        return super().load_detail(start, end)

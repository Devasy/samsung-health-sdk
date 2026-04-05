"""SamsungHealthParser — main entry point for loading health metrics."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from samsung_health_sdk.exercise_analysis import RunAnalysis

import pandas as pd

from samsung_health_sdk.exceptions import MetricNotFoundError
from samsung_health_sdk.metrics.base import BaseMetric
from samsung_health_sdk.metrics.exercise import ExerciseMetric
from samsung_health_sdk.metrics.heart_rate import HeartRateMetric
from samsung_health_sdk.metrics.hrv import HRVMetric
from samsung_health_sdk.metrics.movement import MovementMetric
from samsung_health_sdk.metrics.respiratory_rate import RespiratoryRateMetric
from samsung_health_sdk.metrics.skin_temperature import SkinTemperatureMetric
from samsung_health_sdk.metrics.sleep import SleepStageMetric
from samsung_health_sdk.metrics.spo2 import SpO2Metric
from samsung_health_sdk.metrics.steps import StepsMetric
from samsung_health_sdk.metrics.stress import StressMetric
from samsung_health_sdk.utils import DateLike

# Regex to extract the metric name from a CSV filename
# e.g. "com.samsung.shealth.tracker.heart_rate.20250630001879.csv"
#       → "com.samsung.shealth.tracker.heart_rate"
_CSV_NAME_RE = re.compile(r"^(com\.samsung\..+?)\.\d{10,}\.csv$")


class SamsungHealthParser:
    """
    Parse a Samsung Health export directory.

    Usage::

        from samsung_health_sdk import SamsungHealthParser

        p = SamsungHealthParser("path/to/samsunghealth_export_dir")
        print(p.list_metrics())

        hr = p.get_heart_rate("2024-10-01", "2024-10-31")
        sleep = p.get_sleep("2024-10-01", "2024-10-31")
    """

    def __init__(self, data_dir: str | Path) -> None:
        self._data_dir = Path(data_dir).resolve()
        if not self._data_dir.is_dir():
            raise FileNotFoundError(f"Export directory not found: {self._data_dir}")

        # Build {metric_name: csv_path} map (no file reading, just stat)
        self._metric_map: dict[str, Path] = {}
        for csv_path in self._data_dir.glob("*.csv"):
            m = _CSV_NAME_RE.match(csv_path.name)
            if m:
                self._metric_map[m.group(1)] = csv_path

        # Cache for BaseMetric instances
        self._metric_instances: dict[str, BaseMetric] = {}

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_metrics(self) -> list[str]:
        """Return sorted list of all available metric names."""
        return sorted(self._metric_map.keys())

    def has_metric(self, metric: str) -> bool:
        """Return True if the given metric is present in this export."""
        return metric in self._metric_map

    # ------------------------------------------------------------------
    # Generic accessor
    # ------------------------------------------------------------------

    def get_metric(
        self,
        metric: str,
        start: DateLike = None,
        end: DateLike = None,
        load_binning: bool = False,
    ) -> pd.DataFrame:
        """
        Load any metric by its full name (e.g. 'com.samsung.shealth.stress').

        Parameters
        ----------
        metric:
            Full metric name as returned by list_metrics().
        start, end:
            Optional date bounds ("YYYY-MM-DD", datetime, or pd.Timestamp).
        load_binning:
            If True, expand binning JSON files for minute/second-level detail.

        Raises
        ------
        MetricNotFoundError
            If the metric is not present in this export.
        """
        if metric not in self._metric_map:
            raise MetricNotFoundError(metric, self.list_metrics())

        # Use a known typed metric class if one exists, else generic BaseMetric
        instance = self._get_or_create_metric(metric)
        if load_binning:
            return instance.load_detail(start, end)
        return instance.load_summary(start, end)

    def _get_or_create_metric(self, metric: str) -> BaseMetric:
        if metric not in self._metric_instances:
            # Check if a typed class covers this metric
            typed_classes: list[type[BaseMetric]] = [
                HeartRateMetric,
                SleepStageMetric,
                SkinTemperatureMetric,
                StressMetric,
                SpO2Metric,
                HRVMetric,
                StepsMetric,
                RespiratoryRateMetric,
                ExerciseMetric,
                MovementMetric,
            ]
            for cls in typed_classes:
                if cls.metric_name == metric:
                    self._metric_instances[metric] = cls(self._data_dir)
                    break
            else:
                # Generic fallback
                instance = BaseMetric.__new__(BaseMetric)
                instance.metric_name = metric  # type: ignore[assignment]
                instance.value_columns = []  # type: ignore[assignment]
                instance._data_dir = self._data_dir
                instance._csv_path = self._metric_map[metric]
                instance._summary_cache = None
                self._metric_instances[metric] = instance
        return self._metric_instances[metric]

    # ------------------------------------------------------------------
    # Typed convenience methods
    # ------------------------------------------------------------------

    def get_heart_rate(
        self,
        start: DateLike = None,
        end: DateLike = None,
        granularity: Literal["summary", "minute"] = "summary",
    ) -> pd.DataFrame:
        """
        Heart rate data.

        granularity='summary': one row per ~1-hour session (avg, min, max bpm).
        granularity='minute':  one row per minute from binning JSON files.
        """
        m = HeartRateMetric(self._data_dir)
        if granularity == "minute":
            return m.load_detail(start, end)
        return m.load_summary(start, end)

    def get_sleep(
        self,
        start: DateLike = None,
        end: DateLike = None,
    ) -> pd.DataFrame:
        """
        Sleep stage data.

        Each row is a contiguous sleep stage segment with:
        start_time, end_time, stage (int), stage_label ("Awake"/"Light"/"Deep"/"REM"),
        sleep_id (session grouping).
        """
        return SleepStageMetric(self._data_dir).load_summary(start, end)

    def get_skin_temperature(
        self,
        start: DateLike = None,
        end: DateLike = None,
        granularity: Literal["summary", "minute"] = "summary",
    ) -> pd.DataFrame:
        """
        Skin temperature data (°C).

        granularity='summary': session-level mean/min/max.
        granularity='minute':  per-minute readings from binning JSON files.
        """
        m = SkinTemperatureMetric(self._data_dir)
        if granularity == "minute":
            return m.load_detail(start, end)
        return m.load_summary(start, end)

    def get_stress(
        self,
        start: DateLike = None,
        end: DateLike = None,
    ) -> pd.DataFrame:
        """
        Stress data.

        Columns: start_time, end_time, score (0–100), stress_min, stress_max.
        """
        return StressMetric(self._data_dir).load_summary(start, end)

    def get_spo2(
        self,
        start: DateLike = None,
        end: DateLike = None,
    ) -> pd.DataFrame:
        """
        Blood oxygen saturation (SpO2, %).
        """
        return SpO2Metric(self._data_dir).load_summary(start, end)

    def get_hrv(
        self,
        start: DateLike = None,
        end: DateLike = None,
        load_binning: bool = True,
    ) -> pd.DataFrame:
        """
        Heart Rate Variability (HRV) data.

        load_binning=True (default) expands session JSON files for detailed metrics
        (SDNN, RMSSD, etc. if present in the export).
        """
        m = HRVMetric(self._data_dir)
        if load_binning:
            return m.load_detail(start, end)
        return m.load_summary(start, end)

    def get_steps(
        self,
        start: DateLike = None,
        end: DateLike = None,
    ) -> pd.DataFrame:
        """
        Step count data.

        Each row is ~1 minute of activity with count, walk_step, run_step,
        distance (m), calorie, and speed.
        """
        return StepsMetric(self._data_dir).load_summary(start, end)

    def get_respiratory_rate(
        self,
        start: DateLike = None,
        end: DateLike = None,
        granularity: Literal["summary", "minute"] = "summary",
    ) -> pd.DataFrame:
        """
        Respiratory rate (breaths/min).

        granularity='summary': session average/limits.
        granularity='minute':  per-minute values from binning JSON files.
        """
        m = RespiratoryRateMetric(self._data_dir)
        if granularity == "minute":
            return m.load_detail(start, end)
        return m.load_summary(start, end)

    def get_exercise(
        self,
        start: DateLike = None,
        end: DateLike = None,
    ) -> pd.DataFrame:
        """
        Exercise sessions.

        Columns include: start_time, end_time, exercise_type, exercise_name,
        duration_sec, duration_min, distance_km, speed_kmh, pace_min_per_km,
        calorie, mean_heart_rate, max_heart_rate, mean_speed, vo2_max,
        altitude_gain, altitude_loss, beats_per_km.
        """
        return ExerciseMetric(self._data_dir).load_summary(start, end)

    def get_runs(
        self,
        start: DateLike = None,
        end: DateLike = None,
        min_distance_km: float = 0.5,
    ) -> pd.DataFrame:
        """
        Outdoor running sessions only (exercise_type 1002), deduplicated.

        Duplicate records from companion apps (e.g. Google Fit) are removed;
        the Samsung Health record with the richest HR data is kept.

        Columns include: start_time, end_time, duration_min, distance_km,
        pace_min_per_km, speed_kmh, mean_heart_rate, max_heart_rate,
        mean_cadence, vo2_max, beats_per_km, calorie, altitude_gain, datauuid.
        """
        return ExerciseMetric(self._data_dir).load_runs(start, end, min_distance_km=min_distance_km)

    def get_run_analysis(self, tz: str = "Asia/Kolkata") -> "RunAnalysis":
        """
        Return a :class:`RunAnalysis` instance for this export.

        The RunAnalysis object provides cross-run comparison, per-run time
        series (HR, pace, cadence, beats_per_m), HR zones, and the
        beats_per_km efficiency trend.

        Parameters
        ----------
        tz : str
            Timezone for display formatting (default ``"Asia/Kolkata"`` = IST).

        Example
        -------
        ::

            ra    = parser.get_run_analysis()
            table = ra.compare_runs("2026-01-01", "2026-12-31")
            ts    = ra.run_timeseries(table.loc[0, "datauuid"])
            zones = ra.hr_zones(table.loc[0, "datauuid"])
            trend = ra.beats_per_km_trend("2026-01-01")
        """
        from samsung_health_sdk.exercise_analysis import RunAnalysis

        return RunAnalysis(self, tz=tz)

    def get_movement(
        self,
        start: DateLike = None,
        end: DateLike = None,
    ) -> pd.DataFrame:
        """
        Per-minute movement intensity (activity_level) from accelerometer data.

        Available from Nov 2024 onwards (Galaxy Watch 4+ or compatible device).
        Each row is one 60-second bin. Columns: start_time, end_time,
        activity_level (accelerometer-derived intensity, 0–800+).

        Typical ranges:
            0–5    sedentary   5–20   light
            20–50  low-moderate  50–100  moderate  100+  vigorous
        """
        return MovementMetric(self._data_dir).load_detail(start, end)

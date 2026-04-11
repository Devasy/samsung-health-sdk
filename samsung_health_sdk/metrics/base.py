"""BaseMetric: shared CSV reading, binning loading, and date filtering."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pandas as pd

from samsung_health_sdk.utils import (
    DateLike,
    filter_date_range,
    load_binning_json,
    parse_timestamps,
    read_csv,
    resolve_binning_path,
)


class BaseMetric:
    """
    Base class for all Samsung Health metric parsers.

    Subclasses must set:
        metric_name   - the com.samsung.* metric identifier
        value_columns - list of measurement columns to retain in output
                        (empty list = keep all columns)

    Lazy loading: the CSV is read only on first access and cached.
    """

    metric_name: ClassVar[str] = ""
    value_columns: ClassVar[list[str]] = []

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._csv_path: Path | None = self._find_csv()
        self._summary_cache: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_csv(self) -> Path | None:
        """
        Locate the CSV for this metric by glob (handles the timestamp suffix).

        Samsung Health exports follow the naming convention:
            ``<metric_name>.<timestamp>.csv``
        where ``<timestamp>`` is a sequence of digits (≥ 10 characters).

        The loose glob ``<metric_name>.*.csv`` would also match sub-metrics
        (e.g. ``com.samsung.shealth.exercise.recovery_heart_rate.*``), so we
        filter to only files where the stem stripped of the trailing timestamp
        exactly equals ``metric_name``.
        """
        pattern = f"{self.metric_name}.*.csv"
        matches = []
        for p in self._data_dir.glob(pattern):
            # stem = "com.samsung.shealth.exercise.20260328222493"
            stem_parts = p.stem.rsplit(".", 1)
            if (
                len(stem_parts) == 2
                and stem_parts[1].isdigit()
                and stem_parts[0] == self.metric_name
            ):
                matches.append(p)
        if not matches:
            return None
        # Take the most recently modified if multiple exports exist
        return max(matches, key=lambda p: p.stat().st_mtime)

    def _load_raw(self) -> pd.DataFrame:
        """Read and cache the raw CSV (all rows, timestamps parsed)."""
        if self._summary_cache is not None:
            return self._summary_cache
        if self._csv_path is None:
            return pd.DataFrame()
        df = read_csv(self._csv_path)
        df = parse_timestamps(df)
        self._summary_cache = df
        return df

    def _select_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only value_columns if specified; always keep timing/ID columns."""
        if not self.value_columns or df.empty:
            return df
        keep = set(self.value_columns) | {
            "start_time",
            "end_time",
            "datauuid",
            "time_offset",
            "deviceuuid",
        }
        present = [c for c in df.columns if c in keep]
        return df[present]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """True if the metric CSV exists in this export."""
        return self._csv_path is not None

    def load_summary(
        self,
        start: DateLike = None,
        end: DateLike = None,
    ) -> pd.DataFrame:
        """
        Return the CSV-level (aggregated) data as a DataFrame.

        Each row typically represents one measurement session or hourly/daily summary.
        """
        df = self._load_raw()
        df = filter_date_range(df, start, end)
        return self._select_columns(df)

    def load_detail(
        self,
        start: DateLike = None,
        end: DateLike = None,
    ) -> pd.DataFrame:
        """
        Return minute/second-level data by expanding binning JSON files.

        Falls back to load_summary if no 'binning_data' column is present.
        Only JSON files whose parent session falls within [start, end] are read.
        """
        summary = self._load_raw()
        summary = filter_date_range(summary, start, end)

        if "binning_data" not in summary.columns or summary.empty:
            return self._select_columns(summary)

        chunks: list[pd.DataFrame] = []
        for _, row in summary.iterrows():
            fname = row.get("binning_data")
            if not fname or pd.isna(fname):
                continue
            json_path = resolve_binning_path(self._data_dir, self.metric_name, str(fname))
            detail = load_binning_json(json_path)
            if detail.empty:
                continue
            # Attach parent session metadata
            for meta_col in ("datauuid", "deviceuuid", "time_offset"):
                if meta_col in row.index and meta_col not in detail.columns:
                    detail[meta_col] = row[meta_col]
            chunks.append(detail)

        if not chunks:
            return pd.DataFrame()

        result = pd.concat(chunks, ignore_index=True)
        result = filter_date_range(result, start, end)
        result = result.sort_values("start_time").reset_index(drop=True)
        return result

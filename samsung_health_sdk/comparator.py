"""SamsungHealthComparator — compare health data across multiple people or time windows."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from samsung_health_sdk.parser import SamsungHealthParser
from samsung_health_sdk.utils import DateLike


class SamsungHealthComparator:
    """
    Compare health metrics across multiple people (or the same person across time windows).

    Usage::

        from samsung_health_sdk import SamsungHealthParser, SamsungHealthComparator

        p1 = SamsungHealthParser("path/to/person1_export")
        p2 = SamsungHealthParser("path/to/person2_export")
        comp = SamsungHealthComparator({"Alice": p1, "Bob": p2})

        df = comp.compare_heart_rate("2024-10-01", "2024-10-31")
        # df columns: person, start_time, heart_rate, heart_rate_min, heart_rate_max

        # Align to relative Day 0 per person
        df = comp.compare_heart_rate("2024-10-01", "2024-10-31", time_shift=True)
    """

    def __init__(self, parsers: dict[str, SamsungHealthParser]) -> None:
        if not parsers:
            raise ValueError("At least one parser must be provided.")
        self._parsers = parsers

    @property
    def persons(self) -> list[str]:
        """Return the list of person names."""
        return list(self._parsers.keys())

    # ------------------------------------------------------------------
    # Generic comparison
    # ------------------------------------------------------------------

    def compare_metric(
        self,
        metric: str,
        start: DateLike = None,
        end: DateLike = None,
        persons: list[str] | None = None,
        time_shift: bool = False,
        load_binning: bool = False,
    ) -> pd.DataFrame:
        """
        Load a metric for multiple people and return a tidy long-format DataFrame.

        Parameters
        ----------
        metric:
            Full metric name (e.g. 'com.samsung.shealth.stress').
        start, end:
            Date bounds applied independently to each person's data.
        persons:
            Subset of person names to include. None = all.
        time_shift:
            If True, subtract each person's earliest start_time so that all
            time series begin at timedelta 0 (relative comparison).
        load_binning:
            Expand binning JSON files for minute-level detail.

        Returns
        -------
        pd.DataFrame with a 'person' column prepended.
        """
        selected = persons or list(self._parsers.keys())
        chunks: list[pd.DataFrame] = []

        for name in selected:
            parser = self._parsers.get(name)
            if parser is None:
                raise KeyError(f"Person '{name}' not found. Available: {self.persons}")
            df = parser.get_metric(metric, start=start, end=end, load_binning=load_binning)
            if df.empty:
                continue
            df = df.copy()
            df.insert(0, "person", name)
            chunks.append(df)

        if not chunks:
            return pd.DataFrame()

        result = pd.concat(chunks, ignore_index=True)

        if time_shift and "start_time" in result.columns:
            result = self._apply_time_shift(result)

        return result.sort_values(["person", "start_time"]).reset_index(drop=True)

    @staticmethod
    def _apply_time_shift(df: pd.DataFrame) -> pd.DataFrame:
        """
        Shift each person's timestamps so their minimum start_time becomes t=0.

        Adds a 'relative_time' column (pd.Timedelta from each person's day-0).
        The original 'start_time' column is preserved.
        """
        df = df.copy()
        offsets: dict[str, pd.Timestamp] = {}
        for person, grp in df.groupby("person"):
            offsets[person] = grp["start_time"].min()

        df["relative_time"] = df.apply(
            lambda row: row["start_time"] - offsets[row["person"]], axis=1
        )
        return df

    # ------------------------------------------------------------------
    # Typed convenience wrappers
    # ------------------------------------------------------------------

    def compare_heart_rate(
        self,
        start: DateLike = None,
        end: DateLike = None,
        granularity: Literal["summary", "minute"] = "summary",
        persons: list[str] | None = None,
        time_shift: bool = False,
    ) -> pd.DataFrame:
        """Compare heart rate across persons. See compare_metric for parameter docs."""
        selected = persons or list(self._parsers.keys())
        chunks: list[pd.DataFrame] = []
        for name in selected:
            df = self._parsers[name].get_heart_rate(start, end, granularity=granularity)
            if df.empty:
                continue
            df = df.copy()
            df.insert(0, "person", name)
            chunks.append(df)
        if not chunks:
            return pd.DataFrame()
        result = pd.concat(chunks, ignore_index=True)
        if time_shift and "start_time" in result.columns:
            result = self._apply_time_shift(result)
        return result.sort_values(["person", "start_time"]).reset_index(drop=True)

    def compare_sleep(
        self,
        start: DateLike = None,
        end: DateLike = None,
        persons: list[str] | None = None,
        time_shift: bool = False,
    ) -> pd.DataFrame:
        """Compare sleep stages across persons."""
        selected = persons or list(self._parsers.keys())
        chunks: list[pd.DataFrame] = []
        for name in selected:
            df = self._parsers[name].get_sleep(start, end)
            if df.empty:
                continue
            df = df.copy()
            df.insert(0, "person", name)
            chunks.append(df)
        if not chunks:
            return pd.DataFrame()
        result = pd.concat(chunks, ignore_index=True)
        if time_shift and "start_time" in result.columns:
            result = self._apply_time_shift(result)
        return result.sort_values(["person", "start_time"]).reset_index(drop=True)

    def compare_stress(
        self,
        start: DateLike = None,
        end: DateLike = None,
        persons: list[str] | None = None,
        time_shift: bool = False,
    ) -> pd.DataFrame:
        """Compare stress scores across persons."""
        selected = persons or list(self._parsers.keys())
        chunks: list[pd.DataFrame] = []
        for name in selected:
            df = self._parsers[name].get_stress(start, end)
            if df.empty:
                continue
            df = df.copy()
            df.insert(0, "person", name)
            chunks.append(df)
        if not chunks:
            return pd.DataFrame()
        result = pd.concat(chunks, ignore_index=True)
        if time_shift and "start_time" in result.columns:
            result = self._apply_time_shift(result)
        return result.sort_values(["person", "start_time"]).reset_index(drop=True)

    def compare_steps(
        self,
        start: DateLike = None,
        end: DateLike = None,
        persons: list[str] | None = None,
        time_shift: bool = False,
    ) -> pd.DataFrame:
        """Compare step counts across persons."""
        selected = persons or list(self._parsers.keys())
        chunks: list[pd.DataFrame] = []
        for name in selected:
            df = self._parsers[name].get_steps(start, end)
            if df.empty:
                continue
            df = df.copy()
            df.insert(0, "person", name)
            chunks.append(df)
        if not chunks:
            return pd.DataFrame()
        result = pd.concat(chunks, ignore_index=True)
        if time_shift and "start_time" in result.columns:
            result = self._apply_time_shift(result)
        return result.sort_values(["person", "start_time"]).reset_index(drop=True)

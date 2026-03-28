"""
ReportBuilder — collects all HealthFeatureEngine outputs and renders
them into a single self-contained HTML dashboard file.
"""

from __future__ import annotations

import datetime
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from samsung_health_sdk.features import HealthFeatureEngine

_TEMPLATE = Path(__file__).parent / "template.html"


# ---------------------------------------------------------------------------
# JSON serialisation helpers
# ---------------------------------------------------------------------------

class _Enc(json.JSONEncoder):
    """Handles numpy scalars, datetime.date, NaN → null."""

    def default(self, obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, np.bool_):
            return bool(obj)
        if obj is pd.NaT:
            return None
        return super().default(obj)


def _to_records(df: pd.DataFrame, keep: list[str] | None = None) -> list[dict]:
    """Convert a DataFrame to a JSON-safe list of dicts."""
    if df is None or df.empty:
        return []
    if keep:
        df = df[[c for c in keep if c in df.columns]]
    df = df.copy()
    # Convert bare datetime.date values (object columns) to ISO strings
    for col in df.columns:
        if df[col].dtype == object:
            sample = next((v for v in df[col] if v is not None), None)
            if isinstance(sample, datetime.date) and not isinstance(sample, datetime.datetime):
                df[col] = df[col].apply(
                    lambda v: v.isoformat() if isinstance(v, datetime.date) else v
                )
    # Replace pandas NA / numpy NaN with None so json.dumps emits null
    df = df.where(pd.notna(df), other=None)
    return json.loads(json.dumps(df.to_dict(orient="records"), cls=_Enc))


# ---------------------------------------------------------------------------
# ReportBuilder
# ---------------------------------------------------------------------------

class ReportBuilder:
    """
    Runs all HealthFeatureEngine features over a date range and renders
    the results into a self-contained HTML report.

    Usage::

        from samsung_health_sdk.report import ReportBuilder
        builder = ReportBuilder(engine)
        builder.build("report.html", start="2024-11-01", end="2025-06-30")

    Or via the engine convenience method::

        engine.export_report("report.html", start="2024-11-01", end="2025-06-30")
    """

    def __init__(self, engine: "HealthFeatureEngine") -> None:
        self._eng = engine

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        output_path: str | Path = "report.html",
        start=None,
        end=None,
        title: str = "Samsung Health Dashboard",
    ) -> Path:
        """
        Generate the HTML report and write it to *output_path*.

        Parameters
        ----------
        output_path:
            Destination file path for the generated HTML.
        start, end:
            Date range (str "YYYY-MM-DD", datetime, or None for all data).
        title:
            Dashboard heading shown in the report.

        Returns
        -------
        Path to the written HTML file.
        """
        output_path = Path(output_path)
        data = self._collect(start, end, title)
        json_str = json.dumps(data, cls=_Enc, separators=(",", ":"))
        # Escape </script> to prevent early tag close
        json_str = json_str.replace("</script>", r"<\/script>")

        template = _TEMPLATE.read_text(encoding="utf-8")
        placeholder = "/*__HEALTH_DATA__*/null/**/"
        if placeholder not in template:
            raise ValueError(
                "template.html is missing the __HEALTH_DATA__ placeholder. "
                "The template may have been manually edited."
            )
        html = template.replace(placeholder, f"/*__HEALTH_DATA__*/{json_str}/**/")
        output_path.write_text(html, encoding="utf-8")
        return output_path

    # ------------------------------------------------------------------
    # Internal: data collection
    # ------------------------------------------------------------------

    def _safe(self, fn, *args, **kwargs):
        """Call a feature method; return empty DataFrame on any error."""
        try:
            return fn(*args, **kwargs)
        except Exception:
            return pd.DataFrame()

    def _collect(self, start, end, title: str) -> dict:
        eng = self._eng

        sleep       = self._safe(eng.sleep_sessions,        start, end)
        physio      = self._safe(eng.nightly_physiology,    start, end)
        readiness   = self._safe(eng.hrv_readiness,         start, end)
        stress_sl   = self._safe(eng.stress_impact_on_sleep, start, end)
        activity    = self._safe(eng.daily_activity_profile, start, end)
        cardiac     = self._safe(eng.walking_cardiac_load,   start, end, source="auto")
        hr_daily    = self._safe(eng.daily_hr_stats,         start, end)

        # Fix readiness baseline column to a known name
        if not readiness.empty:
            base_col = next((c for c in readiness.columns if c.startswith("baseline_")), None)
            if base_col and base_col != "baseline_14d":
                readiness = readiness.rename(columns={base_col: "baseline_14d"})

        # Split cardiac into per-bout scatter + deduplicated daily rolling line
        cardiac_bouts, cardiac_rolling = self._split_cardiac(cardiac)

        # Cardiac trend scalar for KPI card
        cardiac_trend = None
        if not cardiac.empty and "cardiac_load_trend" in cardiac.columns:
            valid = cardiac[cardiac["cardiac_load_trend"].notna()]
            if not valid.empty:
                cardiac_trend = str(valid["cardiac_load_trend"].iloc[-1])

        # Date range for header
        all_dates = self._all_dates(sleep, readiness, activity)

        return {
            "meta": {
                "title":          title,
                "generated_at":   datetime.datetime.now().isoformat(timespec="seconds"),
                "date_range": {
                    "start": min(all_dates) if all_dates else None,
                    "end":   max(all_dates) if all_dates else None,
                },
                "cardiac_trend": cardiac_trend,
            },
            "sleep":          _to_records(sleep),
            "physiology":     _to_records(physio),
            "readiness":      _to_records(readiness),
            "stress_sleep":   _to_records(stress_sl),
            "activity":       _to_records(activity),
            "cardiac_bouts":  cardiac_bouts,
            "cardiac_rolling": cardiac_rolling,
            "hr_daily":       _to_records(hr_daily),
        }

    @staticmethod
    def _split_cardiac(cardiac: pd.DataFrame):
        if cardiac.empty:
            return [], []

        roll_col = next(
            (c for c in cardiac.columns if c.startswith("rolling_") and "cardiac" in c),
            None,
        )

        # Per-bout records (drop the rolling col to keep payload small)
        bout_cols = ["date", "duration_min", "distance_m", "speed_mps",
                     "mean_hr", "cardiac_load", "source"]
        bouts = _to_records(cardiac, keep=[c for c in bout_cols if c in cardiac.columns])

        # Deduplicated daily rolling series
        if roll_col:
            rolling_df = (
                cardiac[["date", roll_col]]
                .dropna(subset=[roll_col])
                .drop_duplicates("date")
                .rename(columns={roll_col: "rolling_cardiac_load"})
            )
            rolling = _to_records(rolling_df)
        else:
            rolling = []

        return bouts, rolling

    @staticmethod
    def _all_dates(sleep, readiness, activity) -> list[str]:
        dates: list[str] = []
        for df in (sleep, readiness, activity):
            if df.empty or "date" not in df.columns:
                continue
            for v in df["date"]:
                if v is not None:
                    dates.append(str(v)[:10])
        return sorted(set(dates))

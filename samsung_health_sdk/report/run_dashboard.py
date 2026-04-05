"""
RunDashboardBuilder — generates a self-contained HTML running dashboard.

Usage::

    from samsung_health_sdk import SamsungHealthParser
    from samsung_health_sdk.report.run_dashboard import RunDashboardBuilder

    p = SamsungHealthParser("path/to/export")
    RunDashboardBuilder(p).build("run_dashboard.html", start="2026-01-01")
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
    from samsung_health_sdk.parser import SamsungHealthParser

_TEMPLATE = Path(__file__).parent / "run_dashboard.html"

# Max live-data points to embed per run (downsampled)
_MAX_TS_POINTS = 300


# ---------------------------------------------------------------------------
# JSON helpers (reuse the same pattern as builder.py)
# ---------------------------------------------------------------------------


class _Enc(json.JSONEncoder):
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
    if df is None or df.empty:
        return []
    if keep:
        df = df[[c for c in keep if c in df.columns]]
    df = df.copy().where(pd.notna(df), other=None)
    return json.loads(json.dumps(df.to_dict(orient="records"), cls=_Enc))


def _val(v):
    """Scalar to JSON-safe value."""
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


# ---------------------------------------------------------------------------
# RunDashboardBuilder
# ---------------------------------------------------------------------------


class RunDashboardBuilder:
    """
    Build a self-contained HTML running dashboard from a Samsung Health export.

    Parameters
    ----------
    parser : SamsungHealthParser
        Loaded parser for the export directory.
    tz : str
        Timezone for display labels (default IST).
    """

    def __init__(self, parser: SamsungHealthParser, tz: str = "Asia/Kolkata") -> None:
        self._parser = parser
        self._tz = tz

    def build(
        self,
        output_path: str | Path = "run_dashboard.html",
        start=None,
        end=None,
        title: str = "Running Dashboard",
    ) -> Path:
        """
        Collect run data, inject it into the HTML template, and write the file.

        Returns the path to the written HTML file.
        """
        output_path = Path(output_path)
        data = self._collect(start, end, title)
        json_str = json.dumps(data, cls=_Enc, separators=(",", ":"))
        json_str = json_str.replace("</script>", r"<\/script>")

        template = _TEMPLATE.read_text(encoding="utf-8")
        placeholder = "/*__RUN_DATA__*/null/**/"
        html = template.replace(placeholder, f"/*__RUN_DATA__*/{json_str}/**/")
        output_path.write_text(html, encoding="utf-8")
        return output_path

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def _collect(self, start, end, title: str) -> dict:
        from samsung_health_sdk.exercise_analysis import RunAnalysis

        ra = RunAnalysis(self._parser, tz=self._tz)

        runs_df = ra.compare_runs(start, end)
        trend_df = ra.beats_per_km_trend(start, end, min_distance_km=2.0)

        run_records = _to_records(runs_df)
        trend_records = _to_records(trend_df)

        # Build per-run live data (HR zones, pace breakdown, timeseries)
        live: dict[str, dict] = {}
        for _, row in runs_df.iterrows():
            uuid = row.get("datauuid")
            if not uuid:
                continue
            # Only runs where we have HR data are worth showing live detail
            if pd.isna(row.get("mean_hr")):
                continue
            live[uuid] = self._build_live(ra, uuid, row)

        # Summary KPIs
        all_runs = runs_df
        kpis = {
            "total_runs": len(all_runs),
            "total_distance_km": _val(round(all_runs["distance_km"].sum(), 1))
            if not all_runs.empty
            else None,
            "best_pace": all_runs["pace"].min() if not all_runs.empty else "-",
            "avg_beats_per_km": _val(round(all_runs["beats_per_km"].dropna().mean(), 0))
            if not all_runs.empty
            else None,
            "best_run_uuid": str(all_runs.loc[all_runs["distance_km"].idxmax(), "datauuid"])
            if not all_runs.empty
            else None,
        }

        return {
            "meta": {
                "title": title,
                "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "tz": self._tz,
            },
            "kpis": kpis,
            "runs": run_records,
            "trend": trend_records,
            "live": live,
        }

    def _build_live(self, ra, uuid: str, row) -> dict:
        # HR zones
        zones_df = ra.hr_zones(uuid)
        zones = _to_records(zones_df)

        # Pace breakdown
        pb_df = ra.pace_breakdown(uuid, bucket_min=5)
        pace_breakdown = _to_records(pb_df)

        # Time series — downsample to ≤ _MAX_TS_POINTS
        ts_df = ra.run_timeseries(uuid, smooth_sec=30)
        if not ts_df.empty:
            step = max(1, len(ts_df) // _MAX_TS_POINTS)
            ts_down = ts_df.iloc[::step].copy()
            ts_records = _to_records(
                ts_down,
                keep=[
                    "elapsed_min",
                    "heart_rate",
                    "speed_kmh",
                    "beats_per_m_smooth",
                    "distance_cumulative_km",
                ],
            )
        else:
            ts_records = []

        return {
            "zones": zones,
            "pace_breakdown": pace_breakdown,
            "timeseries": ts_records,
        }

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

from samsung_health_sdk.exercise_analysis import _fmt_pace

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

        trend_records = _to_records(trend_df)

        # Build per-run live data (HR zones, pace breakdown, timeseries, GPS)
        live: dict[str, dict] = {}
        gap_pace_map: dict[str, str] = {}
        for _, row in runs_df.iterrows():
            uuid = row.get("datauuid")
            if not uuid:
                continue
            live[uuid] = self._build_live(ra, uuid, row)
            gap_pace_map[uuid] = live[uuid].get("gap_pace", "–")

        # Inject gap_pace into the runs records
        if not runs_df.empty:
            runs_df = runs_df.copy()
            runs_df["gap_pace"] = runs_df["datauuid"].map(gap_pace_map).fillna("–")

        run_records = _to_records(runs_df)

        # Summary KPIs
        all_runs = runs_df
        kpis = {
            "total_runs": len(all_runs),
            "total_distance_km": _val(round(all_runs["distance_km"].sum(), 1))
            if not all_runs.empty
            else None,
            "best_pace": _fmt_pace(all_runs["pace_min_per_km"].min())
            if not all_runs.empty and "pace_min_per_km" in all_runs.columns
            else "-",
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
            ts_vis = ts_df.copy()

            # Interpolate sparse sensor streams before downsampling
            _interp_cols = [
                "heart_rate",
                "speed_kmh",
                "cadence",
                "distance_cumulative_km",
                "altitude_m",
                "grade_pct",
                "gap_speed_kmh",
                "latitude",
                "longitude",
            ]
            for col in _interp_cols:
                if col in ts_vis.columns:
                    ts_vis[col] = pd.to_numeric(ts_vis[col], errors="coerce").interpolate(
                        limit_direction="both"
                    )

            # Recompute beats_per_m_smooth from interpolated HR and speed so
            # NaN gaps in HR don't propagate into the chart.
            if "heart_rate" in ts_vis.columns and "speed_kmh" in ts_vis.columns:
                safe_speed_ms = (ts_vis["speed_kmh"] / 3.6).replace(0, float("nan"))
                raw_bpm = ts_vis["heart_rate"] / (60.0 * safe_speed_ms)
                ts_vis["beats_per_m_smooth"] = raw_bpm.rolling(
                    window=30, min_periods=1, center=True
                ).mean()

            keep_cols = [
                "elapsed_sec",
                "heart_rate",
                "speed_kmh",
                "cadence",
                "beats_per_m_smooth",
                "distance_cumulative_km",
                "altitude_m",
                "grade_pct",
                "gap_speed_kmh",
                "latitude",
                "longitude",
            ]
            keep_present = [c for c in keep_cols if c in ts_vis.columns]

            if len(ts_vis) > _MAX_TS_POINTS and keep_present:
                bin_ids = np.floor(np.linspace(0, _MAX_TS_POINTS - 1, len(ts_vis))).astype(int)
                ts_down = ts_vis[keep_present].groupby(bin_ids).mean().reset_index(drop=True)
            else:
                ts_down = ts_vis[keep_present].copy() if keep_present else ts_vis.copy()

            # Round stored values to clean decimal noise from bucket averaging
            _round_map = {
                "elapsed_sec": 0,
                "heart_rate": 1,
                "speed_kmh": 2,
                "cadence": 1,
                "altitude_m": 1,
                "grade_pct": 2,
                "gap_speed_kmh": 2,
                "beats_per_m_smooth": 4,
                "distance_cumulative_km": 4,
                "latitude": 6,
                "longitude": 6,
            }
            for col, dp in _round_map.items():
                if col in ts_down.columns:
                    ts_down[col] = ts_down[col].round(dp)
            if "elapsed_sec" in ts_down.columns:
                ts_down["elapsed_sec"] = ts_down["elapsed_sec"].fillna(0).astype(int)

            ts_records = _to_records(ts_down, keep=keep_cols)

            # GPS track — raw 500-point downsample of location data for the map
            gps_track: list = []
            if "latitude" in ts_vis.columns and "longitude" in ts_vis.columns:
                gps_df = ts_vis[["latitude", "longitude"]].dropna()
                if not gps_df.empty:
                    step = max(1, len(gps_df) // 500)
                    gps_track = gps_df.iloc[::step].values.round(6).tolist()
        else:
            ts_records = []
            gps_track = []

        # Compute run-level GAP: median of gap_speed_kmh samples (excludes stops)
        gap_pace_str = "–"
        if ts_records:
            gap_speeds = [
                r["gap_speed_kmh"]
                for r in ts_records
                if r.get("gap_speed_kmh") is not None and r["gap_speed_kmh"] > 0.5
            ]
            if gap_speeds:
                import statistics

                med_gap = statistics.median(gap_speeds)
                # Convert km/h → pace string "M:SS /km"
                mpk = 60.0 / med_gap
                m = int(mpk)
                s = round((mpk - m) * 60)
                if s == 60:
                    m += 1
                    s = 0
                gap_pace_str = f"{m}:{s:02d} /km"

        return {
            "zones": zones,
            "pace_breakdown": pace_breakdown,
            "timeseries": ts_records,
            "gps_track": gps_track,
            "gap_pace": gap_pace_str,
        }

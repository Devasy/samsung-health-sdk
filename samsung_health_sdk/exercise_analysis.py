"""
RunAnalysis — per-run and cross-run analysis for outdoor running sessions.

Features
--------
compare_runs()        Side-by-side comparison table for multiple runs:
                      pace, speed, HR, cadence, calories, beats_per_km.

run_timeseries()      Per-second live-data for one run: speed, HR, cadence,
                      and the ``beats_per_m`` efficiency signal.

hr_zones()            Heart-rate zone distribution (% time) for one run.

beats_per_km_trend()  beats_per_km over all runs — tracks aerobic efficiency
                      improvement over weeks/months.

Usage::

    from samsung_health_sdk import SamsungHealthParser, RunAnalysis

    p   = SamsungHealthParser("path/to/export")
    ra  = RunAnalysis(p)

    # Compare all 2026 runs
    table = ra.compare_runs("2026-01-01", "2026-12-31")

    # Per-second time series for a specific run
    ts = ra.run_timeseries("8990e71f-618a-4402-a20d-17f2455ecde2")

    # HR zones for that run
    zones = ra.hr_zones("8990e71f-618a-4402-a20d-17f2455ecde2")

    # Efficiency trend
    trend = ra.beats_per_km_trend("2026-01-01")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from samsung_health_sdk.utils import DateLike

if TYPE_CHECKING:
    from samsung_health_sdk.parser import SamsungHealthParser

# Default HR zone boundaries as a fraction of max HR
# (Zone 1 = recovery … Zone 5 = max effort)
_HR_ZONE_EDGES = [0.0, 0.60, 0.70, 0.80, 0.90, 1.0, float("inf")]
_HR_ZONE_LABELS = ["Z1 <60%", "Z2 60-70%", "Z3 70-80%", "Z4 80-90%", "Z5 >90%"]

# Smoothing window for beats_per_m rolling average (seconds)
_BPM_SMOOTH_SEC = 30


class RunAnalysis:
    """
    Cross-run and within-run analysis for outdoor running sessions.

    Parameters
    ----------
    parser : SamsungHealthParser
        Loaded parser for a Samsung Health export directory.
    tz : str
        Timezone label used when formatting dates in comparison tables.
        Defaults to 'Asia/Kolkata' (IST).
    """

    def __init__(self, parser: SamsungHealthParser, tz: str = "Asia/Kolkata") -> None:
        from samsung_health_sdk.metrics.exercise import ExerciseMetric

        self._parser = parser
        self._tz = tz
        self._metric = ExerciseMetric(parser._data_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compare_runs(
        self,
        start: DateLike = None,
        end: DateLike = None,
        min_distance_km: float = 0.5,
    ) -> pd.DataFrame:
        """
        Side-by-side comparison table for all running sessions in [start, end].

        Returns
        -------
        pd.DataFrame
            One row per run with columns:

            date            — local date + time (formatted, tz-aware)
            duration_min    — total run duration in minutes
            distance_km     — total distance in km
            pace            — average pace formatted as "M:SS /km"
            speed_kmh       — average speed in km/h
            mean_hr         — mean heart rate (bpm)
            max_hr          — peak heart rate (bpm)
            mean_cadence    — average step cadence (spm)
            vo2_max         — VO2-max estimate from watch (ml/kg/min)
            beats_per_km    — total heartbeats per km (efficiency index;
                              lower = more aerobic / efficient)
            calories        — total active calories (kcal)
            altitude_gain_m — elevation gain (m)
        """
        runs = self._metric.load_runs(start, end, min_distance_km=min_distance_km)
        if runs.empty:
            return pd.DataFrame()

        out = pd.DataFrame()
        out["date"] = runs["start_time"].dt.tz_convert(self._tz).dt.strftime("%Y-%m-%d %H:%M")
        out["duration_min"] = runs["duration_min"].round(1)
        out["distance_km"] = runs["distance_km"].round(2)
        out["pace"] = runs["pace_min_per_km"].map(_fmt_pace)
        out["pace_min_per_km"] = runs["pace_min_per_km"].round(4)
        out["speed_kmh"] = runs["speed_kmh"].round(2)
        out["mean_hr"] = runs["mean_heart_rate"].round(0)
        out["max_hr"] = runs["max_heart_rate"].round(0)
        out["mean_cadence"] = runs["mean_cadence"].round(1)
        out["vo2_max"] = runs["vo2_max"].round(1)
        out["beats_per_km"] = runs["beats_per_km"].round(0)
        out["calories"] = runs["calorie"].round(0)
        out["altitude_gain_m"] = runs["altitude_gain"].round(1)
        out["datauuid"] = runs["datauuid"].values

        return out.reset_index(drop=True)

    def run_timeseries(
        self,
        datauuid: str,
        smooth_sec: int = _BPM_SMOOTH_SEC,
    ) -> pd.DataFrame:
        """
        Per-second live-data time series for a single run.

        Parameters
        ----------
        datauuid : str
            Run identifier from ``compare_runs()["datauuid"]``.
        smooth_sec : int
            Window (seconds) for rolling-average smoothing of ``beats_per_m``.
            Set to 0 or 1 to disable smoothing.

        Returns
        -------
        pd.DataFrame
            Columns:
            start_time, elapsed_sec, elapsed_min,
            heart_rate, speed_kmh, cadence,
            beats_per_m, beats_per_m_smooth  (rolling avg),
            distance_cumulative_km            (cumulative km),
            altitude_m, grade_pct, gap_speed_kmh  (from GPS if available),
            latitude, longitude               (from GPS if available),
            percent_of_vo2max
        """
        df = self._metric.load_run_livedata(datauuid)
        if df.empty:
            return df

        if "mean_speed" in df.columns:
            df = df.rename(columns={"mean_speed": "speed"})

        if "speed" in df.columns:
            df["speed_kmh"] = df["speed"] * 3.6

        # Cumulative distance (m → km)
        if "distance" in df.columns:
            df["distance_cumulative_km"] = df["distance"].cumsum() / 1000.0

        # Smoothed beats_per_m
        if "beats_per_m" in df.columns and smooth_sec > 1:
            df["beats_per_m_smooth"] = (
                df["beats_per_m"].rolling(window=smooth_sec, min_periods=1, center=True).mean()
            )
        elif "beats_per_m" in df.columns:
            df["beats_per_m_smooth"] = df["beats_per_m"]

        # Merge GPS data: latitude, longitude (and optionally altitude) → grade → Grade-Adjusted Pace
        try:
            loc_df = self._metric.load_run_locationdata(datauuid)
            if not loc_df.empty:
                loc_cols = ["start_time", "latitude", "longitude"]
                if "altitude" in loc_df.columns:
                    loc_cols.append("altitude")
                loc_sub = loc_df[loc_cols].dropna(subset=["latitude", "longitude"]).copy()
                df = pd.merge_asof(
                    df.sort_values("start_time"),
                    loc_sub.sort_values("start_time"),
                    on="start_time",
                    direction="nearest",
                    tolerance=pd.Timedelta("5s"),
                )

                if "altitude" in df.columns:
                    # Smooth raw GPS altitude (15-sample window removes spike noise)
                    df["altitude_m"] = (
                        pd.to_numeric(df["altitude"], errors="coerce")
                        .rolling(window=15, min_periods=1, center=True)
                        .mean()
                    )
                    df = df.drop(columns=["altitude"], errors="ignore")

                # Grade = Δaltitude / Δdistance; smooth over 30 samples
                if "altitude_m" in df.columns and "distance_cumulative_km" in df.columns:
                    dist_m = df["distance_cumulative_km"] * 1000.0
                    d_alt = df["altitude_m"].diff().fillna(0)
                    d_dist = dist_m.diff().fillna(0)
                    raw_grade = (d_alt / d_dist.replace(0, float("nan"))).clip(-0.40, 0.40)
                    df["grade_pct"] = (
                        raw_grade.rolling(window=30, min_periods=1, center=True).mean() * 100
                    )

                    # Grade-Adjusted Pace via Minetti et al. (2002) metabolic cost model:
                    #   C(g) = 155.4g⁵ − 30.4g⁴ − 43.3g³ + 46.3g² + 19.5g + 3.6
                    # GAP_speed = actual_speed × C(g) / C(0)   where C(0) = 3.6
                    # This tells you the equivalent flat speed for the same metabolic effort.
                    if "speed_kmh" in df.columns:
                        g = df["grade_pct"].fillna(0) / 100.0
                        Cr = (
                            155.4 * g**5 - 30.4 * g**4 - 43.3 * g**3 + 46.3 * g**2 + 19.5 * g + 3.6
                        ).clip(1.0, 15.0)
                        df["gap_speed_kmh"] = (
                            (df["speed_kmh"] * Cr / 3.6)
                            .rolling(window=30, min_periods=1, center=True)
                            .mean()
                        )
        except Exception:
            # GPS data is optional — never crash the timeseries over it
            pass

        keep = [
            c
            for c in [
                "start_time",
                "elapsed_sec",
                "elapsed_min",
                "heart_rate",
                "speed_kmh",
                "cadence",
                "beats_per_m",
                "beats_per_m_smooth",
                "distance_cumulative_km",
                "altitude_m",
                "grade_pct",
                "gap_speed_kmh",
                "latitude",
                "longitude",
                "percent_of_vo2max",
            ]
            if c in df.columns
        ]
        return df[keep]

    def hr_zones(
        self,
        datauuid: str,
        max_hr: float | None = None,
    ) -> pd.DataFrame:
        """
        Heart-rate zone distribution for a single run.

        Uses the 5-zone model (based on % of max HR):
            Z1  <60%   — recovery / very easy
            Z2  60-70% — aerobic base
            Z3  70-80% — tempo / moderate
            Z4  80-90% — threshold
            Z5  >90%   — VO2-max / sprint

        Parameters
        ----------
        datauuid : str
            Run identifier.
        max_hr : float, optional
            Override the session max HR.  If None, uses the recorded max HR
            from the session summary row (``max_heart_rate`` column).  If that
            value is missing or NaN, falls back to ``ts["heart_rate"].max()``
            — the peak value observed in the live heart-rate timeseries.

        Returns
        -------
        pd.DataFrame
            Columns: zone, bpm_range, samples, pct_time
        """
        ts = self._metric.load_run_livedata(datauuid)
        if ts.empty or "heart_rate" not in ts.columns:
            return pd.DataFrame()

        if max_hr is None:
            runs = self._metric.load_runs()
            row = runs[runs["datauuid"] == datauuid]
            max_hr = float(row["max_heart_rate"].iloc[0]) if not row.empty else None
        if max_hr is None or pd.isna(max_hr):
            max_hr = ts["heart_rate"].max()

        hr = ts["heart_rate"].dropna()
        rows = []
        for i, label in enumerate(_HR_ZONE_LABELS):
            lo = _HR_ZONE_EDGES[i] * max_hr
            hi = _HR_ZONE_EDGES[i + 1] * max_hr
            lo_bpm = round(lo)
            hi_bpm = round(hi) if hi != float("inf") else "max"
            mask = (hr >= lo) & (hr < hi)
            samples = int(mask.sum())
            rows.append(
                {
                    "zone": label,
                    "bpm_range": f"{lo_bpm}–{hi_bpm}",
                    "samples": samples,
                    "pct_time": round(samples / len(hr) * 100, 1) if len(hr) > 0 else 0.0,
                }
            )
        return pd.DataFrame(rows)

    def beats_per_km_trend(
        self,
        start: DateLike = None,
        end: DateLike = None,
        min_distance_km: float = 2.0,
        rolling_weeks: int = 4,
    ) -> pd.DataFrame:
        """
        ``beats_per_km`` over time — tracks aerobic efficiency improvement.

        A falling trend means your heart beats fewer times per km, indicating
        improved aerobic fitness (analogous to a declining cardiac cost).

        Parameters
        ----------
        start, end : DateLike
            Date bounds.
        min_distance_km : float
            Only include runs at least this long (avoids skewing short strides).
        rolling_weeks : int
            Window for the rolling mean overlay.  Set to 0 to omit.

        Returns
        -------
        pd.DataFrame
            Columns: date, distance_km, duration_min, pace, beats_per_km,
                     beats_per_km_rolling  (rolling mean over ``rolling_weeks`` weeks)
        """
        runs = self._metric.load_runs(start, end, min_distance_km=min_distance_km)
        if runs.empty:
            return pd.DataFrame()

        out = pd.DataFrame()
        out["start_time"] = runs["start_time"].dt.tz_convert(self._tz)
        out["date"] = out["start_time"].dt.strftime("%Y-%m-%d")
        out["distance_km"] = runs["distance_km"].round(2)
        out["duration_min"] = runs["duration_min"].round(1)
        out["pace"] = runs["pace_min_per_km"].map(_fmt_pace)
        out["mean_hr"] = runs["mean_heart_rate"].round(0)
        out["beats_per_km"] = runs["beats_per_km"].round(0)

        if rolling_weeks > 0:
            days = max(int(rolling_weeks * 7), 1)
            out_sorted = out.sort_values("start_time").set_index("start_time")
            rolling_vals = (
                out_sorted["beats_per_km"].rolling(window=f"{days}D", min_periods=1).mean().round(0)
            )
            rolling_vals.index = out_sorted.index
            out["beats_per_km_rolling"] = rolling_vals.values

        return out.drop(columns=["start_time"]).reset_index(drop=True)

    def pace_breakdown(
        self,
        datauuid: str,
        bucket_min: float = 5.0,
    ) -> pd.DataFrame:
        """
        Average pace and HR broken down into time-buckets within a single run.

        Parameters
        ----------
        datauuid : str
            Run identifier.
        bucket_min : float
            Bucket size in minutes (default 5).

        Returns
        -------
        pd.DataFrame
            Columns: segment (e.g. "0-5 min"), pace, speed_kmh, mean_hr
        """
        ts = self._metric.load_run_livedata(datauuid)
        if ts.empty:
            return pd.DataFrame()

        if "speed" not in ts.columns and "speed_kmh" in ts.columns:
            ts["speed"] = ts["speed_kmh"] / 3.6
        elif "speed" not in ts.columns:
            return pd.DataFrame()

        if "elapsed_min" not in ts.columns:
            return pd.DataFrame()

        total_min = ts["elapsed_min"].max()
        edges = list(np.arange(0, total_min + bucket_min, bucket_min))
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            seg = ts[(ts["elapsed_min"] >= lo) & (ts["elapsed_min"] < hi)]
            sp = seg["speed"].dropna()
            hr_seg = (
                seg["heart_rate"].dropna()
                if "heart_rate" in seg.columns
                else pd.Series(dtype=float)
            )
            if sp.empty:
                continue
            speed_kmh = float(sp.mean() * 3.6)
            pace = 60.0 / speed_kmh if speed_kmh > 0 else float("nan")
            rows.append(
                {
                    "segment": f"{lo:.0f}–{hi:.0f} min",
                    "pace": _fmt_pace(pace),
                    "speed_kmh": round(speed_kmh, 2),
                    "mean_hr": round(float(hr_seg.mean()), 0) if len(hr_seg) > 0 else float("nan"),
                }
            )
        return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _fmt_pace(pace_min_per_km: float) -> str:
    """Format a float pace (min/km) as 'M:SS /km', or '-' if invalid."""
    if pd.isna(pace_min_per_km) or pace_min_per_km <= 0 or pace_min_per_km > 60:
        return "-"
    minutes = int(pace_min_per_km)
    seconds = int(round((pace_min_per_km - minutes) * 60))
    if seconds == 60:
        minutes += 1
        seconds = 0
    return f"{minutes}:{seconds:02d} /km"

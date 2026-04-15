"""
Build a per-day feature matrix from all HealthFeatureEngine outputs.

The resulting DataFrame has one row per calendar date and ~23 columns
covering every available health dimension: sleep quality, HRV, stress,
activity intensity, respiratory physiology, and cardiac fitness.

Typical usage::

    from samsung_health_sdk import SamsungHealthParser
    from samsung_health_sdk.features import HealthFeatureEngine
    from samsung_health_sdk.ml.feature_matrix import build_daily_features

    p   = SamsungHealthParser("path/to/export")
    eng = HealthFeatureEngine(p)
    df  = build_daily_features(eng)
    print(df.tail())
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from samsung_health_sdk.features import HealthFeatureEngine


# Canonical feature columns output by build_daily_features()
FEATURE_COLS: list[str] = [
    # Sleep quality
    "sleep_quality_score", "efficiency_pct", "deep_min", "rem_min",
    "sleep_light_min", "awake_min", "total_h", "fragmentation_index",
    # HRV
    "rmssd_mean", "hrv_readiness_score", "hrv_deviation_pct",
    # Stress
    "mean_stress", "stress_deviation_pct",
    # Activity intensity (minutes per bucket)
    "sedentary_min", "light_activity_min", "low_mod_min",
    "moderate_min", "vigorous_min", "active_min", "mean_hr_active",
    # Nightly physiology
    "rr_mean", "restlessness_score",
    # Cardiac fitness proxy
    "cardiac_load",
]

# Target columns predicted by the ML model
TARGET_COLS: list[str] = [
    "sleep_quality_score",
    "hrv_readiness_score",
    "energy_index",
]


def build_daily_features(engine: "HealthFeatureEngine") -> pd.DataFrame:
    """
    Merge all HealthFeatureEngine outputs into a single per-day DataFrame.

    Each row represents one calendar date. Missing metric data results in
    NaN for that column on that date; the model handles imputation.

    An additional synthetic column ``energy_index`` (0–100) is derived as a
    weighted blend of sleep quality, HRV readiness, and stress elevation.
    This is one of the three prediction targets.

    Parameters
    ----------
    engine : HealthFeatureEngine
        An initialised feature engine backed by a loaded SamsungHealthParser.

    Returns
    -------
    pd.DataFrame
        Index : pd.DatetimeIndex (one entry per calendar date)
        Columns : FEATURE_COLS + ["energy_index"]
    """
    frames: list[pd.DataFrame] = []

    # ── 1. Sleep sessions ─────────────────────────────────────────────
    try:
        sleep = engine.sleep_sessions()
        if not sleep.empty:
            keep = {
                "quality_score": "sleep_quality_score",
                "efficiency_pct": "efficiency_pct",
                "deep_min": "deep_min",
                "rem_min": "rem_min",
                "light_min": "sleep_light_min",
                "awake_min": "awake_min",
                "total_h": "total_h",
                "fragmentation_index": "fragmentation_index",
            }
            avail = {k: v for k, v in keep.items() if k in sleep.columns}
            sl = sleep[["date"] + list(avail.keys())].rename(columns=avail).copy()
            sl["date"] = pd.to_datetime(sl["date"])
            sl = sl.set_index("date")
            frames.append(sl)
    except Exception:
        pass

    # ── 2. HRV readiness ──────────────────────────────────────────────
    try:
        hrv = engine.hrv_readiness()
        if not hrv.empty:
            keep = {
                "rmssd_mean": "rmssd_mean",
                "readiness_score": "hrv_readiness_score",
                "deviation_pct": "hrv_deviation_pct",
            }
            avail = {k: v for k, v in keep.items() if k in hrv.columns}
            hv = hrv[["date"] + list(avail.keys())].rename(columns=avail).copy()
            hv["date"] = pd.to_datetime(hv["date"])
            hv = hv.set_index("date")
            frames.append(hv)
    except Exception:
        pass

    # ── 3. Daily activity profile ──────────────────────────────────────
    try:
        act = engine.daily_activity_profile()
        if not act.empty:
            keep = {
                "sedentary_min": "sedentary_min",
                "light_min": "light_activity_min",   # rename to avoid clash with sleep_light_min
                "low_mod_min": "low_mod_min",
                "moderate_min": "moderate_min",
                "vigorous_min": "vigorous_min",
                "active_min": "active_min",
                "mean_hr_active": "mean_hr_active",
                "mean_stress": "mean_stress",
                "stress_deviation_pct": "stress_deviation_pct",
            }
            avail = {k: v for k, v in keep.items() if k in act.columns}
            ac = act[["date"] + list(avail.keys())].rename(columns=avail).copy()
            ac["date"] = pd.to_datetime(ac["date"])
            ac = ac.set_index("date")
            frames.append(ac)
    except Exception:
        pass

    # ── 4. Nightly physiology ──────────────────────────────────────────
    try:
        phys = engine.nightly_physiology()
        if not phys.empty:
            keep = {
                "rr_mean": "rr_mean",
                "restlessness_score": "restlessness_score",
            }
            avail = {k: v for k, v in keep.items() if k in phys.columns}
            ph = phys[["date"] + list(avail.keys())].rename(columns=avail).copy()
            ph["date"] = pd.to_datetime(ph["date"])
            ph = ph.set_index("date")
            frames.append(ph)
    except Exception:
        pass

    # ── 5. Cardiac load (daily average over all walking bouts) ─────────
    try:
        cardiac = engine.walking_cardiac_load()
        if not cardiac.empty and "cardiac_load" in cardiac.columns:
            cl = (
                cardiac[["date", "cardiac_load"]]
                .assign(date=lambda d: pd.to_datetime(d["date"]))
                .groupby("date")[["cardiac_load"]]
                .mean()
            )
            frames.append(cl)
    except Exception:
        pass

    if not frames:
        return pd.DataFrame()

    # Outer-join all frames on date index
    merged = frames[0].copy()
    for df in frames[1:]:
        merged = merged.join(df, how="outer", rsuffix="_dup")
        dup_cols = [c for c in merged.columns if c.endswith("_dup")]
        merged.drop(columns=dup_cols, inplace=True)

    merged = merged.sort_index()
    merged.index = pd.to_datetime(merged.index)

    # ── 6. Derived: energy_index (0–100) ──────────────────────────────
    # Blend sleep quality + HRV readiness − stress penalty.
    # Falls back to column median where data is missing.
    s = merged.get("sleep_quality_score", pd.Series(dtype=float))
    h = merged.get("hrv_readiness_score", pd.Series(dtype=float))
    sd = merged.get("stress_deviation_pct", pd.Series(0.0, index=merged.index)).fillna(0.0)

    s_filled = s.fillna(s.median() if s.notna().any() else 60.0)
    h_filled = h.fillna(h.median() if h.notna().any() else 60.0)
    # stress_deviation_pct can be negative (below baseline) or positive (above).
    # Clip to [-20, 50] then map to a 0–1 penalty: 0 = low stress, 1 = high.
    stress_penalty = ((sd.clip(-20, 50) + 20) / 70.0) * 100.0  # 0–100

    merged["energy_index"] = (
        0.45 * s_filled + 0.35 * h_filled - 0.20 * stress_penalty
    ).clip(0, 100)

    return merged

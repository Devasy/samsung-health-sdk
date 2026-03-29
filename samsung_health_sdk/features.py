"""
HealthFeatureEngine — derives higher-level features from parsed Samsung Health data.

Features
--------
sleep_sessions()          Per-night: efficiency, deep/REM minutes, fragmentation index,
                          composite quality score.

nightly_physiology()      Per-night (needs Galaxy Watch with HRV/RR sensors, Nov 2024+):
                          sleep RMSSD mean/min/std, respiratory rate mean/std,
                          sleep restlessness score and restless minutes (from movement).

hrv_readiness()           Per-night: today's sleep RMSSD vs rolling N-day median.
                          Flags nights where HRV is suppressed below personal baseline.

stress_impact_on_sleep()  Per-night: previous-day stress *deviation* from personal
                          rolling baseline vs that night's sleep quality.
                          Uses deviation (not absolute score) to handle chronically
                          elevated baselines.

walking_cardiac_load()    Per walking bout: cardiac cost = mean HR / walking speed.
                          Source 'pedometer' joins 1-min pedometer speed with minute HR.
                          Source 'movement' uses accelerometer activity_level with a
                          speed calibration fitted on the pedometer overlap period,
                          extending cardiac load back to Nov 2024.
                          Source 'exercise' uses session summaries (Jun 2022+).
                          Source 'auto'/'combined' uses the best available source per date.
                          Rolling N-week average shows aerobic fitness trend over months.

daily_activity_profile()  Per-day: sedentary/light/moderate/vigorous minute counts,
                          mean HR during active minutes, median HR for the day,
                          mean stress score and deviation from rolling baseline.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd

from samsung_health_sdk.parser import SamsungHealthParser
from samsung_health_sdk.utils import DateLike, filter_date_range

# activity_level bucket edges and labels (used in daily_activity_profile)
_AL_BINS   = [0, 5, 20, 50, 100, float("inf")]
_AL_LABELS = ["sedentary", "light", "low_mod", "moderate", "vigorous"]

# activity_level threshold that indicates intentional walking movement
_WALK_AL_MIN = 20   # below this → sedentary / light fidgeting
_WALK_AL_MAX = 400  # above this → running / artefact


class HealthFeatureEngine:
    """
    Derive higher-level health features from a SamsungHealthParser instance.

    Usage::

        from samsung_health_sdk import SamsungHealthParser
        from samsung_health_sdk.features import HealthFeatureEngine

        p = SamsungHealthParser("path/to/export")
        eng = HealthFeatureEngine(p)

        sleep   = eng.sleep_sessions()
        physio  = eng.nightly_physiology()
        ready   = eng.hrv_readiness()
        impact  = eng.stress_impact_on_sleep()
        cardiac = eng.walking_cardiac_load()
    """

    def __init__(self, parser: SamsungHealthParser, tz_offset_hours: float = 5.5) -> None:
        """
        Parameters
        ----------
        parser:
            Initialised SamsungHealthParser pointing at an export directory.
        tz_offset_hours:
            UTC offset of the user's local timezone in decimal hours.
            Used to assign calendar dates correctly.
            Default 5.5 = India Standard Time (UTC+0530).
        """
        self._p = parser
        self._tz = pd.Timedelta(hours=tz_offset_hours)
        self._mv_cache: pd.DataFrame | None = None          # movement bins (full range)
        self._speed_cal: tuple[float, float] | None = None  # (a, b): speed = exp(b)*al^a

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _local_date(self, ts: pd.Series) -> pd.Series:
        """Convert UTC timestamps to local calendar dates."""
        return (ts + self._tz).dt.date

    @staticmethod
    def _stage_seconds(group: pd.DataFrame, label: str) -> float:
        sg = group[group["stage_label"] == label]
        return (sg["end_time"] - sg["start_time"]).dt.total_seconds().sum()

    def _get_movement_bins(self, start: DateLike = None, end: DateLike = None) -> pd.DataFrame:
        """
        Return per-minute movement data with 'minute' floor column.
        Full dataset is loaded once and cached; date filtering is applied per-call.
        Loading 5k+ JSON files takes ~30 s on first call.
        """
        if self._mv_cache is None:
            mv = self._p.get_movement()
            if mv.empty or "activity_level" not in mv.columns:
                self._mv_cache = pd.DataFrame()
            else:
                mv = mv.copy()
                mv["activity_level"] = pd.to_numeric(mv["activity_level"], errors="coerce")
                mv["minute"] = mv["start_time"].dt.floor("min")
                self._mv_cache = mv
        df = self._mv_cache
        if df.empty:
            return df
        return filter_date_range(df, start, end)

    def _fit_speed_calibration(self) -> tuple[float, float] | None:
        """
        Fit a log-log linear speed model using the pedometer+movement overlap period.

        Model: log(speed) = a * log(activity_level) + b
              → speed_est = exp(b) * activity_level ** a

        Returns (a, b) coefficients, or None if insufficient overlap data.
        """
        if self._speed_cal is not None:
            return self._speed_cal

        try:
            steps = self._p.get_steps()
        except Exception:
            return None

        if steps.empty or "speed" not in steps.columns:
            return None

        steps = steps[steps["speed"].notna() & (steps["speed"] > 0)].copy()
        if steps.empty:
            return None

        steps["minute"] = steps["start_time"].dt.floor("min")

        # Restrict movement bins to the pedometer date range for calibration
        ped_start = steps["start_time"].min()
        ped_end   = steps["start_time"].max()
        mv = self._get_movement_bins(ped_start, ped_end)
        if mv.empty:
            return None

        cal = steps[["minute", "speed"]].merge(
            mv[["minute", "activity_level"]], on="minute", how="inner"
        )
        # Walking range only
        cal = cal[
            cal["activity_level"].between(_WALK_AL_MIN, _WALK_AL_MAX) &
            cal["speed"].between(0.5, 3.0)
        ]
        if len(cal) < 30:
            warnings.warn(
                "Insufficient overlap data for speed calibration "
                f"({len(cal)} points). Movement cardiac load may be inaccurate.",
                stacklevel=3,
            )
            return None

        log_al = np.log(cal["activity_level"].values)
        log_sp = np.log(cal["speed"].values)
        coeffs = np.polyfit(log_al, log_sp, deg=1)  # [a, b]
        self._speed_cal = (float(coeffs[0]), float(coeffs[1]))
        return self._speed_cal

    # ------------------------------------------------------------------
    # 1. Sleep sessions
    # ------------------------------------------------------------------

    def sleep_sessions(
        self,
        start: DateLike = None,
        end: DateLike = None,
        min_hours: float = 2.0,
    ) -> pd.DataFrame:
        """
        Per-sleep-session quality metrics.

        Returns
        -------
        DataFrame with columns:
            sleep_id, date, total_h, efficiency_pct,
            deep_min, rem_min, light_min, awake_min,
            deep_pct, rem_pct,
            fragmentation_index,   # awake-transition count per sleep hour
            quality_score          # 0–100 composite
        """
        raw = self._p.get_sleep(start, end)
        if raw.empty:
            return pd.DataFrame()

        rows = []
        for sid, g in raw.groupby("sleep_id"):
            total_s = (g["end_time"].max() - g["start_time"].min()).total_seconds()
            if total_s < min_hours * 3600:
                continue

            awake_s = self._stage_seconds(g, "Awake")
            light_s = self._stage_seconds(g, "Light")
            deep_s  = self._stage_seconds(g, "Deep")
            rem_s   = self._stage_seconds(g, "REM")
            sleep_s = total_s - awake_s  # actual sleep time

            efficiency = sleep_s / total_s * 100 if total_s > 0 else np.nan

            # Fragmentation: number of distinct Awake segments per sleep hour
            awake_segs = (g["stage_label"] == "Awake").sum()
            frag_index = awake_segs / (total_s / 3600) if total_s > 0 else np.nan

            deep_pct = deep_s  / sleep_s * 100 if sleep_s > 0 else np.nan
            rem_pct  = rem_s   / sleep_s * 100 if sleep_s > 0 else np.nan

            # Quality score (0–100):
            #   40 pts from efficiency (target ≥85%)
            #   30 pts from deep sleep  (target ≥15% of sleep time)
            #   30 pts from REM sleep   (target ≥20% of sleep time)
            q_eff  = min(efficiency / 85, 1.0) * 40  if not np.isnan(efficiency) else 0
            q_deep = min(deep_pct / 15,   1.0) * 30  if not np.isnan(deep_pct)   else 0
            q_rem  = min(rem_pct  / 20,   1.0) * 30  if not np.isnan(rem_pct)    else 0
            quality = q_eff + q_deep + q_rem

            rows.append({
                "sleep_id":           sid,
                "date":               self._local_date(g["start_time"]).iloc[0],
                "total_h":            round(total_s / 3600, 2),
                "efficiency_pct":     round(efficiency, 1),
                "deep_min":           round(deep_s / 60, 1),
                "rem_min":            round(rem_s / 60, 1),
                "light_min":          round(light_s / 60, 1),
                "awake_min":          round(awake_s / 60, 1),
                "deep_pct":           round(deep_pct, 1),
                "rem_pct":            round(rem_pct, 1),
                "fragmentation_index": round(frag_index, 2),
                "quality_score":      round(quality, 1),
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("date").reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # 2. Nightly physiology (HRV + RR during sleep)
    # ------------------------------------------------------------------

    def nightly_physiology(
        self,
        start: DateLike = None,
        end: DateLike = None,
        min_hours: float = 2.0,
    ) -> pd.DataFrame:
        """
        Per-sleep-session HRV and respiratory rate, measured during sleep.

        Only available for nights where the watch recorded HRV/RR data
        (typically Galaxy Watch 4+ from the night the user wore it).

        Returns
        -------
        DataFrame with columns:
            sleep_id, date,
            rmssd_mean, rmssd_min, rmssd_std,   # ms — higher/stable = more parasympathetic
            rr_mean, rr_std,                     # breaths/min
            restlessness_score,                  # mean activity_level during sleep window
            restless_min,                        # minutes with activity_level > 20 (movement)
            hrv_suppression_flag                 # True if rmssd_mean < 0.85 × personal_median
        """
        sleep = self._p.get_sleep(start, end)
        hrv   = self._p.get_hrv(start, end)
        rr    = self._p.get_respiratory_rate(start, end, granularity="minute")
        mv    = self._get_movement_bins(start, end)  # may be empty — optional

        if sleep.empty or hrv.empty:
            return pd.DataFrame()

        sessions = sleep.groupby("sleep_id").agg(
            sleep_start=("start_time", "min"),
            sleep_end=("end_time",   "max"),
        ).reset_index()

        rows = []
        for _, s in sessions.iterrows():
            duration_h = (s.sleep_end - s.sleep_start).total_seconds() / 3600
            if duration_h < min_hours:
                continue

            w_hrv = hrv[
                (hrv["start_time"] >= s.sleep_start) &
                (hrv["start_time"] <= s.sleep_end)
            ]
            w_rr = rr[
                (rr["start_time"] >= s.sleep_start) &
                (rr["start_time"] <= s.sleep_end)
            ] if not rr.empty else pd.DataFrame()

            if w_hrv.empty:
                continue

            rr_valid = (
                w_rr[w_rr["respiratory_rate"] > 0]["respiratory_rate"]
                if not w_rr.empty else pd.Series(dtype=float)
            )

            # Movement restlessness during sleep
            if not mv.empty:
                w_mv = mv[
                    (mv["start_time"] >= s.sleep_start) &
                    (mv["start_time"] <= s.sleep_end)
                ]
                restlessness_score = round(w_mv["activity_level"].mean(), 2) if not w_mv.empty else np.nan
                restless_min       = int((w_mv["activity_level"] > _WALK_AL_MIN).sum()) if not w_mv.empty else 0
            else:
                restlessness_score = np.nan
                restless_min       = np.nan

            rows.append({
                "sleep_id":          s.sleep_id,
                "date":              self._local_date(pd.Series([s.sleep_start])).iloc[0],
                "rmssd_mean":        round(w_hrv["rmssd"].mean(), 2),
                "rmssd_min":         round(w_hrv["rmssd"].min(),  2),
                "rmssd_std":         round(w_hrv["rmssd"].std(),  2),
                "rr_mean":           round(rr_valid.mean(), 2) if len(rr_valid) else np.nan,
                "rr_std":            round(rr_valid.std(),  2) if len(rr_valid) else np.nan,
                "restlessness_score": restlessness_score,
                "restless_min":      restless_min,
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        # Flag nights where RMSSD is suppressed below 85% of personal median
        personal_median = df["rmssd_mean"].median()
        df["hrv_suppression_flag"] = df["rmssd_mean"] < 0.85 * personal_median
        df = df.sort_values("date").reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # 3. HRV readiness
    # ------------------------------------------------------------------

    def hrv_readiness(
        self,
        start: DateLike = None,
        end: DateLike = None,
        baseline_days: int = 14,
    ) -> pd.DataFrame:
        """
        Daily HRV readiness: how today's sleep RMSSD compares to the rolling
        N-day personal baseline.

        A deviation below -15% is flagged as 'low readiness' — it typically
        reflects accumulated stress, illness onset, or insufficient recovery.

        Returns
        -------
        DataFrame with columns:
            date, rmssd_mean,
            baseline_{N}d,           # rolling median over prior N nights
            deviation_pct,           # (today - baseline) / baseline × 100
            readiness_score,         # 0–100 (50 = at baseline, capped)
            low_readiness_flag
        """
        physio = self.nightly_physiology(start=start, end=end)
        if physio.empty:
            return pd.DataFrame()

        physio = physio.sort_values("date").reset_index(drop=True)

        # Rolling median over the prior N nights (shift=1 so today isn't included)
        baseline_col = f"baseline_{baseline_days}d"
        physio[baseline_col] = (
            physio["rmssd_mean"]
            .shift(1)
            .rolling(window=baseline_days, min_periods=max(3, baseline_days // 3))
            .median()
        )

        physio["deviation_pct"] = (
            (physio["rmssd_mean"] - physio[baseline_col])
            / physio[baseline_col] * 100
        ).round(1)

        # Readiness score: 50 at baseline, +/- proportional, capped 0–100
        physio["readiness_score"] = (
            (50 + physio["deviation_pct"] * 0.5).clip(0, 100).round(1)
        )

        physio["low_readiness_flag"] = physio["deviation_pct"] < -15

        cols = ["date", "rmssd_mean", baseline_col,
                "deviation_pct", "readiness_score", "low_readiness_flag"]
        return physio[cols].dropna(subset=[baseline_col]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 4. Stress impact on sleep
    # ------------------------------------------------------------------

    def stress_impact_on_sleep(
        self,
        start: DateLike = None,
        end: DateLike = None,
        stress_baseline_days: int = 30,
        min_hours: float = 2.0,
    ) -> pd.DataFrame:
        """
        For each night, report the *previous day's stress deviation* from the
        user's rolling personal baseline, alongside that night's sleep quality.

        Using deviation (not absolute score) avoids the noise caused by
        chronically elevated baselines — a score of 75 means different things
        depending on whether your typical is 60 or 80.

        Returns
        -------
        DataFrame with columns:
            date, sleep_id,
            prev_stress_mean,       # previous day mean stress score
            prev_stress_baseline,   # rolling N-day median of daily stress
            prev_stress_deviation,  # deviation_pct from baseline (negative = calmer day)
            quality_score, efficiency_pct, deep_min, rem_min,
            total_h, fragmentation_index
        """
        stress = self._p.get_stress(start, end)
        sleep_stats = self.sleep_sessions(start=start, end=end, min_hours=min_hours)

        if stress.empty or sleep_stats.empty:
            return pd.DataFrame()

        # Daily stress: mean per local calendar day
        stress["local_date"] = self._local_date(stress["start_time"])
        daily = stress.groupby("local_date")["score"].mean().reset_index()
        daily.columns = ["date", "stress_mean"]
        daily = daily.sort_values("date").reset_index(drop=True)

        # Rolling baseline (prior N days, not including today)
        daily["stress_baseline"] = (
            daily["stress_mean"]
            .shift(1)
            .rolling(window=stress_baseline_days,
                     min_periods=max(5, stress_baseline_days // 4))
            .median()
        )
        daily["stress_deviation_pct"] = (
            (daily["stress_mean"] - daily["stress_baseline"])
            / daily["stress_baseline"] * 100
        ).round(1)

        # Merge: sleep date → previous calendar day's stress
        sleep_stats["prev_date"] = sleep_stats["date"].apply(
            lambda d: (pd.Timestamp(d) - pd.Timedelta(days=1)).date()
        )
        merged = sleep_stats.merge(
            daily.rename(columns={
                "date":                  "prev_date",
                "stress_mean":           "prev_stress_mean",
                "stress_baseline":       "prev_stress_baseline",
                "stress_deviation_pct":  "prev_stress_deviation",
            }),
            on="prev_date",
            how="inner",
        )

        cols = [
            "date", "sleep_id",
            "prev_stress_mean", "prev_stress_baseline", "prev_stress_deviation",
            "quality_score", "efficiency_pct", "deep_min", "rem_min",
            "total_h", "fragmentation_index",
        ]
        return merged[cols].dropna(subset=["prev_stress_baseline"]).sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 5. Walking cardiac load
    # ------------------------------------------------------------------

    def _load_exercise_walks(
        self,
        start: DateLike,
        end: DateLike,
        min_duration_sec: float,
        min_distance_m: float,
    ) -> pd.DataFrame:
        """Load walking bouts from exercise session summaries (Jun 2022+)."""
        ex = self._p.get_exercise(start, end)
        if ex.empty:
            return pd.DataFrame()

        walks = ex[
            (ex["exercise_name"] == "Walking") &
            ex["mean_heart_rate"].notna() &
            ex["distance"].notna() &
            ex["duration_sec"].notna() &
            (ex["duration_sec"] >= min_duration_sec) &
            (ex["distance"] >= min_distance_m)
        ].copy()

        if walks.empty:
            return pd.DataFrame()

        walks["date"]         = self._local_date(walks["start_time"])
        walks["duration_min"] = (walks["duration_sec"] / 60).round(1)
        walks["speed_mps"]    = (walks["distance"] / walks["duration_sec"]).round(3)
        walks["mean_hr"]      = walks["mean_heart_rate"]
        walks["cardiac_load"] = (walks["mean_hr"] / walks["speed_mps"]).round(2)
        walks["source"]       = "exercise"

        walks = walks[
            walks["speed_mps"].between(0.3, 3.0) &
            walks["cardiac_load"].between(30, 300)
        ]

        return walks[["date", "duration_min", "distance", "speed_mps",
                       "mean_hr", "cardiac_load", "source"]].rename(
            columns={"distance": "distance_m"}
        ).reset_index(drop=True)

    def _load_pedometer_walks(
        self,
        start: DateLike,
        end: DateLike,
        min_duration_sec: float,
        min_distance_m: float,
    ) -> pd.DataFrame:
        """
        Load walking bouts from pedometer (1-min resolution) joined with minute HR.

        Pedometer data is available from May 2025 on devices that record it.
        Each row is one 60-second bin; rows with speed > 0 indicate walking.
        Consecutive walking minutes (gap ≤ 2 min) are grouped into bouts.
        """
        try:
            steps = self._p.get_steps(start, end)
            hr_min = self._p.get_heart_rate(start, end, granularity="minute")
        except Exception:
            return pd.DataFrame()

        if steps.empty or hr_min.empty:
            return pd.DataFrame()

        # Only keep walking minutes
        steps = steps[steps["speed"].notna() & (steps["speed"] > 0)].copy()
        if steps.empty:
            return pd.DataFrame()

        # Floor both to minute precision for joining
        steps["minute"] = steps["start_time"].dt.floor("min")
        hr_min = hr_min.copy()
        hr_min["minute"] = hr_min["start_time"].dt.floor("min")

        # Keep a single HR value per minute (mean in case of duplicates)
        hr_per_min = (
            hr_min.groupby("minute")["heart_rate"].mean().reset_index()
        )

        joined = steps.merge(hr_per_min, on="minute", how="inner")
        if joined.empty:
            return pd.DataFrame()

        # Filter implausible HR readings
        joined = joined[joined["heart_rate"].between(40, 220)]
        if joined.empty:
            return pd.DataFrame()

        joined = joined.sort_values("minute").reset_index(drop=True)

        # Group consecutive walking minutes into bouts (gap ≤ 2 min)
        gap_minutes = (joined["minute"].diff().dt.total_seconds() / 60).fillna(0)
        joined["bout_id"] = (gap_minutes > 2).cumsum()

        rows = []
        for bid, bout in joined.groupby("bout_id"):
            dur_sec = len(bout) * 60  # each row is exactly 60 s
            dist_m  = bout["distance"].sum() if "distance" in bout.columns else np.nan
            if dur_sec < min_duration_sec:
                continue
            if not np.isnan(dist_m) and dist_m < min_distance_m:
                continue

            mean_hr    = bout["heart_rate"].mean()
            mean_speed = bout["speed"].mean()
            if mean_speed <= 0:
                continue

            cardiac_load = mean_hr / mean_speed

            rows.append({
                "date":         self._local_date(pd.Series([bout["minute"].iloc[0]])).iloc[0],
                "duration_min": round(dur_sec / 60, 1),
                "distance_m":   round(dist_m, 0) if not np.isnan(dist_m) else np.nan,
                "speed_mps":    round(mean_speed, 3),
                "mean_hr":      round(mean_hr, 1),
                "cardiac_load": round(cardiac_load, 2),
                "source":       "pedometer",
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df[
            df["speed_mps"].between(0.3, 3.0) &
            df["cardiac_load"].between(30, 300)
        ]
        return df.reset_index(drop=True)

    def _load_movement_walks(
        self,
        start: DateLike,
        end: DateLike,
        min_duration_sec: float,
        min_distance_m: float,
    ) -> pd.DataFrame:
        """
        Walking bouts from movement (activity_level) + minute HR.

        Uses a speed model calibrated on the pedometer overlap period to convert
        activity_level → estimated speed, then computes cardiac_load = HR / speed.
        Extends cardiac load coverage back to Nov 2024.
        """
        cal = self._fit_speed_calibration()
        if cal is None:
            return pd.DataFrame()
        a, b = cal

        mv = self._get_movement_bins(start, end)
        if mv.empty:
            return pd.DataFrame()

        # Only keep walking-intensity minutes
        mv_walk = mv[mv["activity_level"].between(_WALK_AL_MIN, _WALK_AL_MAX)].copy()
        if mv_walk.empty:
            return pd.DataFrame()

        # Join with minute HR
        try:
            hr_min = self._p.get_heart_rate(start, end, granularity="minute")
        except Exception:
            return pd.DataFrame()

        if hr_min.empty:
            return pd.DataFrame()

        hr_min = hr_min.copy()
        hr_min["minute"] = hr_min["start_time"].dt.floor("min")
        hr_per_min = hr_min.groupby("minute")["heart_rate"].mean().reset_index()

        joined = mv_walk.merge(hr_per_min, on="minute", how="inner")
        joined = joined[joined["heart_rate"].between(40, 220)]
        if joined.empty:
            return pd.DataFrame()

        joined = joined.sort_values("minute").reset_index(drop=True)

        # Group consecutive walking minutes into bouts (gap ≤ 2 min)
        gap_min = (joined["minute"].diff().dt.total_seconds() / 60).fillna(0)
        joined["bout_id"] = (gap_min > 2).cumsum()

        rows = []
        for _, bout in joined.groupby("bout_id"):
            dur_sec    = len(bout) * 60
            mean_hr    = bout["heart_rate"].mean()
            mean_al    = bout["activity_level"].mean()
            speed_est  = float(np.exp(b) * mean_al ** a)
            speed_est  = max(0.3, min(speed_est, 3.0))

            if dur_sec < min_duration_sec:
                continue

            cardiac_load = mean_hr / speed_est

            rows.append({
                "date":         self._local_date(pd.Series([bout["minute"].iloc[0]])).iloc[0],
                "duration_min": round(dur_sec / 60, 1),
                "distance_m":   np.nan,  # not available from movement data
                "speed_mps":    round(speed_est, 3),
                "mean_hr":      round(mean_hr, 1),
                "cardiac_load": round(cardiac_load, 2),
                "source":       "movement",
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df[df["cardiac_load"].between(30, 300)]
        return df.reset_index(drop=True)

    def walking_cardiac_load(
        self,
        start: DateLike = None,
        end: DateLike = None,
        min_duration_sec: float = 120,
        min_distance_m: float = 100,
        rolling_weeks: int = 4,
        source: Literal["auto", "pedometer", "exercise", "movement", "combined"] = "auto",
    ) -> pd.DataFrame:
        """
        Cardiac cost of walking across sessions, and rolling fitness trend.

        Metric: HR per unit speed (bpm / [m/s]) — lower = heart works less
        hard for the same walking pace = aerobic fitness improvement.

        Speed is normalised so the metric is comparable across sessions of
        different pace.

        Data sources
        ------------
        ``'pedometer'``   1-min pedometer speed joined with minute HR.
                          Fine-grained; captures all walking regardless of workout
                          trigger. Available from May 2025 on supported devices.

        ``'movement'``    Accelerometer activity_level joined with minute HR, with
                          speed estimated via a calibration fit on the pedometer
                          overlap period. Extends coverage back to Nov 2024.

        ``'exercise'``    Exercise session summaries (Walking type only).
                          Available from Jun 2022; requires ≥ ~10 min of continuous
                          walking and a manual/auto workout trigger.

        ``'combined'``    Priority order per date: pedometer > movement > exercise.
                          Gives the fullest historical coverage.

        ``'auto'``        Same as ``'combined'``.

        Returns
        -------
        DataFrame with columns:
            date, duration_min, distance_m, speed_mps,
            mean_hr, cardiac_load,              # mean_hr / speed_mps
            source,                             # 'pedometer' or 'exercise'
            rolling_{N}w_cardiac_load,          # N-week rolling mean
            cardiac_load_trend                  # 'improving' / 'stable' / 'declining'
        """
        kwargs = dict(
            start=start,
            end=end,
            min_duration_sec=min_duration_sec,
            min_distance_m=min_distance_m,
        )

        if source == "pedometer":
            walks = self._load_pedometer_walks(**kwargs)
        elif source == "movement":
            walks = self._load_movement_walks(**kwargs)
        elif source == "exercise":
            walks = self._load_exercise_walks(**kwargs)
        else:  # 'auto' or 'combined': pedometer > movement > exercise
            ped = self._load_pedometer_walks(**kwargs)
            mov = self._load_movement_walks(**kwargs)
            ex  = self._load_exercise_walks(**kwargs)

            # Determine which dates each source covers
            ped_dates = set(ped["date"].unique()) if not ped.empty else set()
            mov_dates = set(mov["date"].unique()) if not mov.empty else set()

            # Exercise only fills dates not covered by pedometer or movement
            ex_fill = ex[~ex["date"].isin(ped_dates | mov_dates)] if not ex.empty else ex
            # Movement fills dates not covered by pedometer
            mov_fill = mov[~mov["date"].isin(ped_dates)] if not mov.empty else mov

            parts = [p for p in (ped, mov_fill, ex_fill) if not p.empty]
            if not parts:
                return pd.DataFrame()
            walks = pd.concat(parts, ignore_index=True)

        if walks.empty:
            return pd.DataFrame()

        walks = walks.sort_values("date").reset_index(drop=True)

        # Rolling N-week mean (date-based)
        walks_daily = (
            walks.groupby("date")[["cardiac_load"]]
            .mean()
            .reset_index()
        )
        roll_col = f"rolling_{rolling_weeks}w_cardiac_load"
        window = rolling_weeks * 7  # days
        walks_daily[roll_col] = (
            walks_daily["cardiac_load"]
            .rolling(window=min(window, len(walks_daily)),
                     min_periods=max(2, rolling_weeks))
            .mean()
            .round(2)
        )

        # Trend: compare last-quarter vs first-quarter of rolling series
        valid = walks_daily[roll_col].dropna()
        if len(valid) >= 8:
            q = len(valid) // 4
            early  = valid.iloc[:q].mean()
            recent = valid.iloc[-q:].mean()
            pct_change = (recent - early) / early * 100
            if pct_change < -5:
                trend = "improving"      # lower cardiac load = fitter
            elif pct_change > 5:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        walks_daily["cardiac_load_trend"] = trend

        result = walks.merge(
            walks_daily[["date", roll_col, "cardiac_load_trend"]],
            on="date", how="left"
        )
        return result.sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 6. Daily activity profile
    # ------------------------------------------------------------------

    def daily_activity_profile(
        self,
        start: DateLike = None,
        end: DateLike = None,
        stress_baseline_days: int = 30,
    ) -> pd.DataFrame:
        """
        Per-day breakdown of activity intensity, heart rate context, and stress.

        Combines per-minute movement data with minute-level HR and daily stress
        to give a holistic picture of each day's physical load.

        Returns
        -------
        DataFrame with columns:
            date,
            sedentary_min,   # activity_level 0–5
            light_min,       # 5–20
            low_mod_min,     # 20–50
            moderate_min,    # 50–100
            vigorous_min,    # 100+
            active_min,      # total non-sedentary (light + above)
            total_tracked_min,
            mean_hr_active,  # mean HR during low_mod/moderate/vigorous minutes
            median_hr_day,   # median HR across all tracked minutes
            mean_stress,     # mean stress score for this day
            stress_baseline, # rolling N-day median (prior days)
            stress_deviation_pct
        """
        mv = self._get_movement_bins(start, end)
        if mv.empty:
            return pd.DataFrame()

        # Minute HR join
        try:
            hr_min = self._p.get_heart_rate(start, end, granularity="minute")
        except Exception:
            hr_min = pd.DataFrame()

        if not hr_min.empty:
            hr_min = hr_min.copy()
            hr_min["minute"] = hr_min["start_time"].dt.floor("min")
            hr_per_min = hr_min.groupby("minute")["heart_rate"].mean().reset_index()
            mv_hr = mv.merge(hr_per_min, on="minute", how="left")
        else:
            mv_hr = mv.copy()
            mv_hr["heart_rate"] = np.nan

        mv_hr["local_date"] = self._local_date(mv_hr["start_time"])
        mv_hr["al_bucket"] = pd.cut(
            mv_hr["activity_level"],
            bins=_AL_BINS,
            labels=_AL_LABELS,
            right=False,
        )

        rows = []
        for date, g in mv_hr.groupby("local_date"):
            counts = g["al_bucket"].value_counts()
            sed_m  = int(counts.get("sedentary", 0))
            li_m   = int(counts.get("light",     0))
            lm_m   = int(counts.get("low_mod",   0))
            mo_m   = int(counts.get("moderate",  0))
            vi_m   = int(counts.get("vigorous",  0))
            active = li_m + lm_m + mo_m + vi_m

            # HR stats
            hr_active = g.loc[g["activity_level"] >= _WALK_AL_MIN, "heart_rate"].dropna()
            hr_all    = g["heart_rate"].dropna()

            rows.append({
                "date":              date,
                "sedentary_min":     sed_m,
                "light_min":         li_m,
                "low_mod_min":       lm_m,
                "moderate_min":      mo_m,
                "vigorous_min":      vi_m,
                "active_min":        active,
                "total_tracked_min": len(g),
                "mean_hr_active":    round(hr_active.mean(), 1) if len(hr_active) else np.nan,
                "median_hr_day":     round(hr_all.median(),  1) if len(hr_all)    else np.nan,
            })

        df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        if df.empty:
            return df

        # Stress context
        try:
            stress = self._p.get_stress(start, end)
        except Exception:
            stress = pd.DataFrame()

        if not stress.empty:
            stress["local_date"] = self._local_date(stress["start_time"])
            daily_stress = stress.groupby("local_date")["score"].mean().reset_index()
            daily_stress.columns = ["date", "mean_stress"]
            daily_stress = daily_stress.sort_values("date").reset_index(drop=True)

            daily_stress["stress_baseline"] = (
                daily_stress["mean_stress"]
                .shift(1)
                .rolling(window=stress_baseline_days,
                         min_periods=max(5, stress_baseline_days // 4))
                .median()
            )
            daily_stress["stress_deviation_pct"] = (
                (daily_stress["mean_stress"] - daily_stress["stress_baseline"])
                / daily_stress["stress_baseline"] * 100
            ).round(1)

            df = df.merge(
                daily_stress[["date", "mean_stress", "stress_baseline", "stress_deviation_pct"]],
                on="date", how="left",
            )
        else:
            df["mean_stress"]           = np.nan
            df["stress_baseline"]       = np.nan
            df["stress_deviation_pct"]  = np.nan

        return df

    # ------------------------------------------------------------------
    # 7. Daily heart-rate percentile statistics
    # ------------------------------------------------------------------

    def daily_hr_stats(
        self,
        start: DateLike = None,
        end: DateLike = None,
    ) -> pd.DataFrame:
        """
        Daily heart-rate percentile statistics.

        Groups session-level HR measurements (≈1 hour each) by calendar day
        and computes distribution metrics.  P5 captures resting/low-activity
        periods; P95 captures peak-activity periods.

        Returns
        -------
        DataFrame with columns:
            date, n_sessions,
            hr_p5, hr_p25, hr_median, hr_p75, hr_p95, hr_mean
        """
        hr = self._p.get_heart_rate(start, end, granularity="summary")
        if hr.empty or "heart_rate" not in hr.columns:
            return pd.DataFrame()

        hr = hr.copy()
        hr["date"] = self._local_date(hr["start_time"])
        hr["heart_rate"] = pd.to_numeric(hr["heart_rate"], errors="coerce")
        hr = hr.dropna(subset=["heart_rate"])

        agg = (
            hr.groupby("date")["heart_rate"]
            .agg(n_sessions="count", hr_mean="mean", hr_median="median")
            .reset_index()
        )
        pcts = (
            hr.groupby("date")["heart_rate"]
            .quantile([0.05, 0.25, 0.75, 0.95])
            .unstack(level=-1)
            .rename(columns={0.05: "hr_p5", 0.25: "hr_p25", 0.75: "hr_p75", 0.95: "hr_p95"})
            .reset_index()
        )
        result = agg.merge(pcts, on="date")
        for col in ["hr_mean", "hr_median", "hr_p5", "hr_p25", "hr_p75", "hr_p95"]:
            result[col] = result[col].round(1)
        return result.sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 8. HTML report export
    # ------------------------------------------------------------------

    def export_report(
        self,
        output_path: str = "report.html",
        start: DateLike = None,
        end: DateLike = None,
        title: str = "Samsung Health Dashboard",
    ) -> Any:
        """
        Generate a self-contained HTML dashboard and write it to *output_path*.

        Runs all feature computations (sleep, HRV, activity, cardiac load, etc.)
        and renders them into an interactive single-file report with ECharts charts.

        Parameters
        ----------
        output_path:
            Destination path for the HTML file (default: ``"report.html"``).
        start, end:
            Date range filter — "YYYY-MM-DD", datetime, or None for all data.
        title:
            Dashboard heading shown in the report.

        Returns
        -------
        pathlib.Path pointing to the written file.

        Example
        -------
        ::

            eng = HealthFeatureEngine(parser)
            eng.export_report("my_health_2025.html",
                               start="2024-11-01", end="2025-06-30")
        """
        from samsung_health_sdk.report.builder import ReportBuilder
        return ReportBuilder(self).build(output_path, start=start, end=end, title=title)

"""
Minute-level signal datasets for cross-signal correlation and anomaly detection.

Two datasets are provided:

MinuteLevelDataset
    Joins heart rate and movement activity_level on minute timestamps.
    Each sample is a sliding window of ``window_size`` minutes with
    activity_level as input features and heart rate as the prediction target.
    Used to train ``MovementHRPredictor``.

SleepWindowDataset
    Extracts multi-signal (HR, movement, respiratory rate) patches from each
    sleep session.  Each sample is a ``window_size``-minute patch over all
    available signals.  Used to train ``SleepMultivariateAE``.

Both datasets expose a ``metadata`` DataFrame that maps each sample index to
its source timestamps, enabling anomaly results to be plotted back on the
original time axis.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from samsung_health_sdk.features import HealthFeatureEngine


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _norm_series(s: pd.Series) -> tuple[pd.Series, float, float]:
    """Min-max normalise a series.  Returns (normed, min_val, max_val)."""
    lo, hi = s.min(), s.max()
    rng = hi - lo if hi != lo else 1.0
    return (s - lo) / rng, float(lo), float(hi)


def _build_minute_grid(
    start: pd.Timestamp,
    end: pd.Timestamp,
    freq: str = "1min",
) -> pd.DatetimeIndex:
    return pd.date_range(start=start.floor("min"), end=end.floor("min"), freq=freq)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Waking HR ↔ Movement dataset
# ──────────────────────────────────────────────────────────────────────────────

class MinuteLevelDataset(Dataset):
    """
    Sliding-window dataset: ``window_size`` minutes of activity_level → HR.

    Each sample:
        x : Tensor (window_size, n_input_features)
            activity_level + optional contextual channels
            (time-of-day sin/cos, day-of-week sin/cos)
        y : Tensor (window_size,)
            heart_rate for the same window (sequence-to-sequence)

    Parameters
    ----------
    engine : HealthFeatureEngine
        Loaded feature engine.
    window_size : int
        Number of consecutive minutes per sample.
    stride : int
        Step size between consecutive windows (1 = fully sliding).
    include_time_features : bool
        Append cyclical time-of-day and day-of-week features to each step.
    exclude_sleep : bool
        If True, drop minutes that fall inside a sleep session so the model
        learns only waking-hour HR/movement relationships.
    """

    def __init__(
        self,
        engine: "HealthFeatureEngine",
        window_size: int = 15,
        stride: int = 5,
        include_time_features: bool = True,
        exclude_sleep: bool = True,
    ) -> None:
        self.window_size = window_size
        self.stride = stride

        joint = _load_waking_joint(engine, exclude_sleep=exclude_sleep)
        if joint.empty:
            raise ValueError("No overlapping HR + movement data found.")

        # Cyclical time features
        if include_time_features:
            mins_in_day = joint.index.hour * 60 + joint.index.minute
            joint["tod_sin"] = np.sin(2 * np.pi * mins_in_day / 1440)
            joint["tod_cos"] = np.cos(2 * np.pi * mins_in_day / 1440)
            dow = joint.index.dayofweek
            joint["dow_sin"] = np.sin(2 * np.pi * dow / 7)
            joint["dow_cos"] = np.cos(2 * np.pi * dow / 7)

        # Normalise signals
        hr_raw = joint["heart_rate"].copy()
        al_raw = joint["activity_level"].copy()
        joint["heart_rate"], self.hr_min, self.hr_max = _norm_series(hr_raw)
        joint["activity_level"], self.al_min, self.al_max = _norm_series(al_raw)

        input_cols = ["activity_level"]
        if include_time_features:
            input_cols += ["tod_sin", "tod_cos", "dow_sin", "dow_cos"]

        X_all = joint[input_cols].values.astype(np.float32)
        y_all = joint["heart_rate"].values.astype(np.float32)
        ts_all = np.array(joint.index)

        # Build windows
        indices = range(0, len(X_all) - window_size + 1, stride)
        self._X   = np.stack([X_all[i: i + window_size] for i in indices])
        self._y   = np.stack([y_all[i: i + window_size] for i in indices])
        self._ts  = [ts_all[i: i + window_size] for i in indices]

        # Metadata for mapping predictions back to timestamps
        self.metadata = pd.DataFrame({
            "sample_idx": range(len(indices)),
            "start_time": [t[0] for t in self._ts],
            "end_time":   [t[-1] for t in self._ts],
        })
        self.input_cols = input_cols
        self.n_input_features = len(input_cols)
        self._raw_joint = joint   # keep for residual analysis

    def __len__(self) -> int:
        return len(self._X)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self._X[idx]), torch.from_numpy(self._y[idx])

    def denorm_hr(self, y_norm: np.ndarray) -> np.ndarray:
        """Inverse-transform normalised HR values to bpm."""
        return y_norm * (self.hr_max - self.hr_min) + self.hr_min

    def get_full_series(self) -> pd.DataFrame:
        """Return the underlying minute-level DataFrame (normalised)."""
        return self._raw_joint


def _load_waking_joint(
    engine: "HealthFeatureEngine",
    exclude_sleep: bool = True,
) -> pd.DataFrame:
    """
    Join minute-level heart rate and movement activity_level.

    Returns a DataFrame indexed by UTC minute timestamp with columns:
        heart_rate, activity_level
    """
    p = engine._p

    # Movement (minute-level)
    mv = engine._get_movement_bins()
    if mv.empty or "activity_level" not in mv.columns:
        raise ValueError("No movement data available.")
    mv = mv[["minute", "activity_level"]].copy()
    mv["activity_level"] = pd.to_numeric(mv["activity_level"], errors="coerce")
    mv = mv.dropna().groupby("minute")["activity_level"].mean().rename("activity_level")

    # Heart rate (minute-level detail)
    try:
        hr_raw = p.get_heart_rate(granularity="detail")
    except Exception:
        hr_raw = p.get_heart_rate()
    if hr_raw.empty or "heart_rate" not in hr_raw.columns:
        raise ValueError("No heart rate detail data available.")
    hr_raw = hr_raw[["start_time", "heart_rate"]].copy()
    hr_raw["heart_rate"] = pd.to_numeric(hr_raw["heart_rate"], errors="coerce")
    hr_raw = hr_raw.dropna()
    hr_raw["minute"] = hr_raw["start_time"].dt.floor("min")
    hr_min = hr_raw.groupby("minute")["heart_rate"].mean().rename("heart_rate")

    # Join
    joint = pd.DataFrame({"heart_rate": hr_min, "activity_level": mv}).dropna()
    joint.index.name = "timestamp"

    # Remove physiologically impossible values
    joint = joint[(joint["heart_rate"] >= 30) & (joint["heart_rate"] <= 220)]
    joint = joint[(joint["activity_level"] >= 0)]

    if exclude_sleep and hasattr(engine, '_p'):
        try:
            sleep_stages = p.get_sleep()
            if not sleep_stages.empty:
                sleep_mask = pd.Series(False, index=joint.index)
                for _, row in sleep_stages.iterrows():
                    mask = (joint.index >= row["start_time"]) & (joint.index <= row["end_time"])
                    sleep_mask |= mask
                joint = joint[~sleep_mask]
        except Exception:
            pass

    return joint.sort_index()


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Sleep multi-signal dataset
# ──────────────────────────────────────────────────────────────────────────────

# Signal columns present in sleep windows (subset actually available varies)
SLEEP_SIGNALS = ["heart_rate", "activity_level", "respiratory_rate"]


class SleepWindowDataset(Dataset):
    """
    Sliding-window dataset of multi-signal sleep patches for autoencoder training.

    Each sample:
        x : Tensor (window_size, n_signals)
            Available signals during a sleep window, normalised to [0, 1].

    The reconstruction target ``y`` is identical to ``x``; the autoencoder
    is trained to reproduce its input and anomalies produce high residuals.

    Parameters
    ----------
    engine : HealthFeatureEngine
        Loaded feature engine.
    window_size : int
        Minutes per patch (default 30).
    stride : int
        Step between patches in minutes (default 5 — heavily overlapping
        to give dense per-minute anomaly scores).
    min_signals : int
        Minimum number of non-missing signals required to include a patch
        in the dataset (default 2).
    """

    def __init__(
        self,
        engine: "HealthFeatureEngine",
        window_size: int = 30,
        stride: int = 5,
        min_signals: int = 2,
    ) -> None:
        self.window_size = window_size
        self.stride = stride

        minute_df, self.norm_params, self.signal_cols = _load_sleep_signals(engine)
        if minute_df.empty:
            raise ValueError("No sleep signal data found.")

        self._minute_df = minute_df

        # Build windows across all sleep sessions
        windows_X:  list[np.ndarray] = []
        windows_ts: list[np.ndarray] = []
        windows_night: list[str] = []

        for night_date, session_df in minute_df.groupby("night_date"):
            arr = session_df[self.signal_cols].values.astype(np.float32)
            ts  = session_df.index.values
            n   = len(arr)
            for i in range(0, n - window_size + 1, stride):
                patch = arr[i: i + window_size]
                # Require at least min_signals columns to have data (not all NaN)
                valid_cols = np.sum(~np.isnan(patch).all(axis=0))
                if valid_cols < min_signals:
                    continue
                # Replace NaN with column mean of the patch
                col_means = np.nanmean(patch, axis=0)
                for j in range(patch.shape[1]):
                    nan_mask = np.isnan(patch[:, j])
                    patch[nan_mask, j] = col_means[j] if not np.isnan(col_means[j]) else 0.5
                windows_X.append(patch)
                windows_ts.append(ts[i: i + window_size])
                windows_night.append(str(night_date))

        if not windows_X:
            raise ValueError(
                "No complete sleep windows could be built. "
                "Check that sleep stage boundaries and HR data overlap."
            )

        self._X = np.stack(windows_X)        # (N, W, C)
        self._ts = windows_ts
        self._nights = windows_night
        self.n_signals = self._X.shape[2]

        self.metadata = pd.DataFrame({
            "sample_idx": range(len(windows_X)),
            "night_date": windows_night,
            "start_time": [t[0] for t in windows_ts],
            "end_time":   [t[-1] for t in windows_ts],
        })

    def __len__(self) -> int:
        return len(self._X)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self._X[idx])
        return x, x   # autoencoder: target = input

    def get_minute_df(self) -> pd.DataFrame:
        """Return the full minute-level sleep DataFrame (all sessions)."""
        return self._minute_df

    def per_minute_errors(
        self,
        recon_errors: np.ndarray,    # (N, window_size) per-step reconstruction error
    ) -> pd.Series:
        """
        Aggregate per-window, per-step reconstruction errors back to a
        minute-resolution time series by averaging across overlapping windows.

        Parameters
        ----------
        recon_errors : np.ndarray (N, window_size)
            Per-step mean-squared reconstruction error for each window.

        Returns
        -------
        pd.Series indexed by UTC timestamp with per-minute anomaly score.
        """
        accum: dict[pd.Timestamp, list[float]] = {}
        for i, ts_arr in enumerate(self._ts):
            for j, ts in enumerate(ts_arr):
                t = pd.Timestamp(ts)
                accum.setdefault(t, []).append(float(recon_errors[i, j]))
        ts_index = sorted(accum.keys())
        scores = [np.mean(accum[t]) for t in ts_index]
        return pd.Series(scores, index=ts_index, name="recon_error")


def _load_sleep_signals(
    engine: "HealthFeatureEngine",
) -> tuple[pd.DataFrame, dict, list[str]]:
    """
    Build a minute-level DataFrame of all available sleep-time signals.

    Returns
    -------
    minute_df : pd.DataFrame
        Indexed by UTC minute timestamp.  Columns: [signal_cols] + ["night_date"].
    norm_params : dict
        {signal: (min, max)} for inverse-transforming predictions.
    signal_cols : list[str]
        Ordered list of signal column names actually present in minute_df.
    """
    p = engine._p

    # ── Sleep session boundaries ──────────────────────────────────────
    try:
        sleep = p.get_sleep()
        if sleep.empty:
            raise ValueError("No sleep sessions.")
    except Exception as exc:
        raise ValueError(f"Cannot load sleep sessions: {exc}") from exc

    # ── Heart rate during sleep ───────────────────────────────────────
    hr_min: pd.Series | None = None
    try:
        hr_raw = p.get_heart_rate(granularity="detail")
        if not hr_raw.empty and "heart_rate" in hr_raw.columns:
            hr_raw = hr_raw[["start_time", "heart_rate"]].copy()
            hr_raw["heart_rate"] = pd.to_numeric(hr_raw["heart_rate"], errors="coerce")
            hr_raw = hr_raw.dropna()
            hr_raw["minute"] = hr_raw["start_time"].dt.floor("min")
            hr_min = hr_raw.groupby("minute")["heart_rate"].mean()
    except Exception:
        pass

    # ── Movement during sleep ─────────────────────────────────────────
    mv_min: pd.Series | None = None
    try:
        mv = engine._get_movement_bins()
        if not mv.empty and "activity_level" in mv.columns:
            mv = mv[["minute", "activity_level"]].copy()
            mv["activity_level"] = pd.to_numeric(mv["activity_level"], errors="coerce")
            mv = mv.dropna()
            mv_min = mv.groupby("minute")["activity_level"].mean()
    except Exception:
        pass

    # ── Respiratory rate during sleep ─────────────────────────────────
    rr_min: pd.Series | None = None
    try:
        rr_raw = p.get_respiratory_rate(granularity="detail")
        if not rr_raw.empty:
            rr_col = "respiratory_rate" if "respiratory_rate" in rr_raw.columns else rr_raw.columns[-1]
            rr_raw = rr_raw[["start_time", rr_col]].copy()
            rr_raw[rr_col] = pd.to_numeric(rr_raw[rr_col], errors="coerce")
            rr_raw = rr_raw.dropna()
            rr_raw["minute"] = rr_raw["start_time"].dt.floor("min")
            rr_min = rr_raw.groupby("minute")[rr_col].mean().rename("respiratory_rate")
    except Exception:
        pass

    if hr_min is None and mv_min is None and rr_min is None:
        raise ValueError("No minute-level sleep signals available.")

    # ── Combine available signals ─────────────────────────────────────
    available: dict[str, pd.Series] = {}
    if hr_min is not None:
        available["heart_rate"] = hr_min
    if mv_min is not None:
        available["activity_level"] = mv_min
    if rr_min is not None:
        available["respiratory_rate"] = rr_min
    signal_cols = list(available.keys())

    combined = pd.DataFrame(available).sort_index()

    # ── Filter to sleep session windows only ──────────────────────────
    sleep_mask = pd.Series(False, index=combined.index)
    night_col  = pd.Series(pd.NaT, index=combined.index, dtype="object")

    for _, row in sleep.iterrows():
        mask = (combined.index >= row["start_time"]) & (combined.index <= row["end_time"])
        sleep_mask |= mask
        # Assign night date = local date of sleep midpoint
        mid = row["start_time"] + (row["end_time"] - row["start_time"]) / 2
        local_mid = mid + engine._tz
        night_col[mask] = str(local_mid.date())

    combined = combined[sleep_mask].copy()
    combined["night_date"] = night_col[sleep_mask].values

    if combined.empty:
        raise ValueError("Sleep time filter removed all signal data.")

    # ── Normalise each signal ──────────────────────────────────────────
    norm_params: dict[str, tuple[float, float]] = {}
    for col in signal_cols:
        lo = float(combined[col].quantile(0.01))  # robust to outliers
        hi = float(combined[col].quantile(0.99))
        rng = hi - lo if hi != lo else 1.0
        combined[col] = ((combined[col] - lo) / rng).clip(0, 1)
        norm_params[col] = (lo, hi)

    return combined, norm_params, signal_cols

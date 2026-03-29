"""
PyTorch Dataset for sliding-window health metric forecasting.

Each sample consists of:

  X : Tensor (seq_len, n_features)
      Feature values for days  [t, t+seq_len-1]  — the look-back window.
      All columns are used as features, including past values of the target
      metrics (past sleep quality, past HRV readiness, …).

  y : Tensor (n_targets,)
      Target values for day  t+seq_len  (the day immediately after the window).
      Targets: [sleep_quality_score, hrv_readiness_score, energy_index]

Both X and y are min-max normalised to [0, 1] using statistics computed from
the full dataset.  The normalisation parameters (feat_min, feat_max, etc.) are
stored on the dataset object so the trainer can save them alongside the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from samsung_health_sdk.ml.feature_matrix import TARGET_COLS


class HealthWindowDataset(Dataset):
    """
    Sliding-window dataset built from a daily feature DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``build_daily_features()``.  Index must be a DatetimeIndex.
    seq_len : int
        Number of past days used as model input (look-back window).
    feature_cols : list[str] | None
        Columns to treat as input features.  If None, all columns in *df*
        are used (including past values of the prediction targets, which is
        valid and informative for autoregressive forecasting).
    augment : bool
        When True, add small Gaussian noise during ``__getitem__``.  Enable
        only on the training split.
    noise_std : float
        Noise standard deviation as a fraction of each feature's range.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = 14,
        feature_cols: list[str] | None = None,
        augment: bool = False,
        noise_std: float = 0.015,
    ) -> None:
        self.seq_len = seq_len
        self.augment = augment
        self.noise_std = noise_std

        # Determine which columns are targets vs features
        self.target_cols = [c for c in TARGET_COLS if c in df.columns]
        if feature_cols is not None:
            self.feature_cols = [c for c in feature_cols if c in df.columns]
        else:
            # Use ALL columns as features (past target values are valid inputs)
            self.feature_cols = list(df.columns)

        all_cols = list(dict.fromkeys(self.feature_cols + self.target_cols))
        data = df[all_cols].copy()

        # Impute NaN with per-column median (forward-fill first for temporal continuity)
        data = data.ffill().bfill()
        for col in data.columns:
            if data[col].isna().any():
                data[col] = data[col].fillna(data[col].median())
                if data[col].isna().all():
                    data[col] = 0.0

        # Min-max normalise features to [0, 1]
        self.feat_min = data[self.feature_cols].min()
        self.feat_max = data[self.feature_cols].max()
        feat_range = (self.feat_max - self.feat_min).replace(0, 1.0)
        data[self.feature_cols] = (data[self.feature_cols] - self.feat_min) / feat_range

        # Min-max normalise targets to [0, 1]
        self.tgt_min = data[self.target_cols].min()
        self.tgt_max = data[self.target_cols].max()
        tgt_range = (self.tgt_max - self.tgt_min).replace(0, 1.0)
        data[self.target_cols] = (data[self.target_cols] - self.tgt_min) / tgt_range

        self._feat = data[self.feature_cols].values.astype(np.float32)
        self._tgt = data[self.target_cols].values.astype(np.float32)
        self.dates = list(df.index)

    def __len__(self) -> int:
        # Need seq_len days for input + 1 day for the target
        return max(0, len(self._feat) - self.seq_len)

    def __getitem__(self, idx: int):
        x = self._feat[idx : idx + self.seq_len].copy()
        y = self._tgt[idx + self.seq_len]

        if self.augment:
            noise = np.random.normal(0, self.noise_std, x.shape).astype(np.float32)
            x = np.clip(x + noise, 0.0, 1.0)

        return torch.from_numpy(x), torch.from_numpy(y)

    def get_window(self, idx: int) -> tuple[torch.Tensor, list]:
        """
        Return the raw (un-augmented) input tensor and corresponding date labels.

        Used by InsightEngine to run inference and recover attention-weight
        interpretations per past day.

        Returns
        -------
        x_tensor : Tensor (seq_len, n_features)
        date_labels : list of date-like objects (length seq_len)
        """
        x = self._feat[idx : idx + self.seq_len].copy()
        dates = self.dates[idx : idx + self.seq_len]
        return torch.from_numpy(x), dates

    def denorm_targets(self, y_norm: np.ndarray) -> np.ndarray:
        """Inverse-transform normalised target values back to original scale."""
        tgt_range = (self.tgt_max - self.tgt_min).replace(0, 1.0).values
        return y_norm * tgt_range + self.tgt_min.values

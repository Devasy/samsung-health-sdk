"""
Cross-signal correlation models for anomaly detection.

Two models
----------

MovementHRPredictor
    Learns the minute-level relationship between movement intensity
    (``activity_level``) and heart rate.  Trained on all waking-hour data,
    it produces per-minute predicted HR.  The residual

        residual = actual_HR − predicted_HR

    is normalised to a rolling z-score.  Periods where |z| > threshold are
    flagged as anomalies:

    * Positive anomaly (HR higher than movement predicts):
      elevated physiological stress, illness, caffeine, emotional arousal.
    * Negative anomaly (HR lower than movement predicts):
      improved aerobic fitness, medication effect, bradycardia.

SleepMultivariateAE
    A sequence-to-sequence LSTM autoencoder trained on 30-minute patches of
    sleep signals (HR, movement, respiratory rate).  It learns the joint
    distribution of these signals during normal sleep.

    Anomaly score per minute = mean reconstruction error across all
    overlapping windows that include that minute.  High scores surface:

    * Night-wide elevation:   illness, fever, overtraining, high stress day
    * Localised spikes:       arousal events, apnea-like episodes, position
                              changes accompanied by HR surges, RR disruptions

SignalAnomalyEngine
    Orchestrator class that wraps both models.  Call ``fit_all(engine)`` to
    train both models in one step, then ``analyse_waking()`` and
    ``analyse_sleep()`` to get anomaly DataFrames ready for plotting.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from samsung_health_sdk.ml.signal_dataset import (
    MinuteLevelDataset,
    SleepWindowDataset,
    SLEEP_SIGNALS,
)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Movement → HR predictor
# ──────────────────────────────────────────────────────────────────────────────

class _MovementHRNet(nn.Module):
    """
    Bidirectional GRU sequence-to-sequence predictor.

    Input:  (batch, window_size, n_input_features)
    Output: (batch, window_size)   — normalised HR at each step
    """

    def __init__(self, n_input: int, hidden: int = 32, n_layers: int = 2,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_input,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)            # (B, T, 2H)
        return self.head(out).squeeze(-1)  # (B, T)


class MovementHRPredictor:
    """
    Train and apply a GRU model that predicts HR from movement + time context.

    Parameters
    ----------
    window_size : int
        Minutes per sliding window (default 15).
    hidden : int
        GRU hidden units per direction.
    lr : float
        Adam learning rate.
    device : str | None
        Auto-detected when None.
    """

    def __init__(
        self,
        window_size: int = 15,
        hidden: int = 32,
        lr: float = 1e-3,
        device: str | None = None,
    ) -> None:
        self.window_size = window_size
        self.hidden = hidden
        self.lr = lr
        self.device = torch.device(_auto_device(device))
        self._net: _MovementHRNet | None = None
        self._ds:  MinuteLevelDataset | None = None

    # ------------------------------------------------------------------

    def fit(
        self,
        ds: MinuteLevelDataset,
        epochs: int = 60,
        batch_size: int = 256,
        val_split: float = 0.15,
        patience: int = 10,
        verbose: bool = True,
    ) -> list[float]:
        """
        Train the predictor on a MinuteLevelDataset.

        Returns training loss history (per epoch).
        """
        self._ds = ds
        n_val   = max(1, int(len(ds) * val_split))
        n_train = len(ds) - n_val
        # Chronological split
        train_ds = Subset(ds, list(range(n_train)))
        val_ds   = Subset(ds, list(range(n_train, len(ds))))

        self._net = _MovementHRNet(
            n_input=ds.n_input_features,
            hidden=self.hidden,
        ).to(self.device)

        optim  = torch.optim.Adam(self._net.parameters(), lr=self.lr, weight_decay=1e-4)
        sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, patience=5, factor=0.5, min_lr=1e-5
        )
        crit   = nn.HuberLoss(delta=0.05)
        best_v = float("inf")
        best_w = None
        no_imp = 0
        history: list[float] = []

        t_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
        v_loader = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

        for epoch in range(1, epochs + 1):
            self._net.train()
            t_losses = []
            for x_b, y_b in t_loader:
                x_b, y_b = x_b.to(self.device), y_b.to(self.device)
                optim.zero_grad()
                pred = self._net(x_b)
                loss = crit(pred, y_b)
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                optim.step()
                t_losses.append(loss.item())

            self._net.eval()
            v_losses = []
            with torch.no_grad():
                for x_b, y_b in v_loader:
                    x_b, y_b = x_b.to(self.device), y_b.to(self.device)
                    v_losses.append(crit(self._net(x_b), y_b).item())

            t_l = float(np.mean(t_losses))
            v_l = float(np.mean(v_losses)) if v_losses else 0.0
            sched.step(v_l)
            history.append(t_l)

            if v_l < best_v:
                best_v = v_l
                best_w = {k: v.cpu().clone() for k, v in self._net.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1

            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"  [HR predictor] epoch {epoch:3d} | train {t_l:.4f} | val {v_l:.4f}")

            if no_imp >= patience:
                if verbose:
                    print(f"  Early stop at epoch {epoch}.")
                break

        if best_w is not None:
            self._net.load_state_dict(best_w)
        return history

    # ------------------------------------------------------------------

    def anomaly_series(
        self,
        z_window: int = 120,
        z_threshold: float = 2.0,
    ) -> pd.DataFrame:
        """
        Compute per-minute HR prediction residuals and anomaly flags.

        Uses a stride-1 pass over the full dataset (all windows) to get
        predicted HR at every minute, then computes rolling z-scores.

        Parameters
        ----------
        z_window : int
            Rolling window (minutes) used to compute μ and σ for z-scoring.
        z_threshold : float
            |z| above this value → anomaly flag.

        Returns
        -------
        pd.DataFrame indexed by UTC timestamp with columns:
            actual_hr, predicted_hr, residual, z_score,
            anomaly_flag, anomaly_direction
            ("elevated" | "depressed" | "normal")
        """
        if self._net is None or self._ds is None:
            raise RuntimeError("Call fit() before anomaly_series().")

        # Stride-1 dataset for full-resolution inference
        full_ds = MinuteLevelDataset.__new__(MinuteLevelDataset)
        full_ds.window_size = self.window_size
        full_ds.stride = 1
        full_ds.hr_min = self._ds.hr_min
        full_ds.hr_max = self._ds.hr_max
        full_ds.al_min = self._ds.al_min
        full_ds.al_max = self._ds.al_max
        full_ds.input_cols = self._ds.input_cols
        full_ds.n_input_features = self._ds.n_input_features
        full_ds._raw_joint = self._ds._raw_joint
        # Rebuild windows with stride=1
        joint = self._ds._raw_joint
        X_all = joint[self._ds.input_cols].values.astype(np.float32)
        y_all = joint["heart_rate"].values.astype(np.float32)
        ts_all = np.array(joint.index)
        ws = self.window_size
        full_ds._X  = np.stack([X_all[i: i + ws] for i in range(len(X_all) - ws + 1)])
        full_ds._y  = np.stack([y_all[i: i + ws] for i in range(len(y_all) - ws + 1)])
        full_ds._ts = [ts_all[i: i + ws] for i in range(len(ts_all) - ws + 1)]

        loader = DataLoader(full_ds, batch_size=512, shuffle=False)
        self._net.eval()
        all_pred: list[np.ndarray] = []
        with torch.no_grad():
            for x_b, _ in loader:
                all_pred.append(self._net(x_b.to(self.device)).cpu().numpy())
        pred_all = np.concatenate(all_pred, axis=0)   # (N, window_size)

        # Average predictions across overlapping windows → per-minute
        accum_pred: dict[pd.Timestamp, list[float]] = {}
        accum_true: dict[pd.Timestamp, list[float]] = {}
        for i, ts_arr in enumerate(full_ds._ts):
            for j, ts in enumerate(ts_arr):
                t = pd.Timestamp(ts)
                accum_pred.setdefault(t, []).append(float(pred_all[i, j]))
                accum_true.setdefault(t, []).append(float(full_ds._y[i, j]))

        ts_sorted = sorted(accum_pred.keys())
        pred_hr_norm = np.array([np.mean(accum_pred[t]) for t in ts_sorted])
        true_hr_norm = np.array([np.mean(accum_true[t]) for t in ts_sorted])

        # Inverse-transform to bpm
        hr_range = self._ds.hr_max - self._ds.hr_min
        pred_bpm = pred_hr_norm * hr_range + self._ds.hr_min
        true_bpm = true_hr_norm * hr_range + self._ds.hr_min

        residuals = true_bpm - pred_bpm

        # Rolling z-score
        res_s = pd.Series(residuals, index=ts_sorted)
        roll_mu  = res_s.rolling(z_window, min_periods=10, center=True).mean()
        roll_sig = res_s.rolling(z_window, min_periods=10, center=True).std().replace(0, 1e-3)
        z_scores = (res_s - roll_mu) / roll_sig

        anomaly_flag = z_scores.abs() > z_threshold
        direction = pd.Series("normal", index=ts_sorted)
        direction[z_scores >  z_threshold] = "elevated"
        direction[z_scores < -z_threshold] = "depressed"

        return pd.DataFrame({
            "actual_hr":        true_bpm,
            "predicted_hr":     pred_bpm,
            "residual":         residuals,
            "z_score":          z_scores.values,
            "anomaly_flag":     anomaly_flag.values,
            "anomaly_direction": direction.values,
        }, index=ts_sorted)

    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        torch.save({
            "net_state":    self._net.state_dict(),
            "net_config":   {
                "n_input":  self._ds.n_input_features,
                "hidden":   self.hidden,
            },
            "window_size":  self.window_size,
            "hr_min":       self._ds.hr_min,
            "hr_max":       self._ds.hr_max,
            "al_min":       self._ds.al_min,
            "al_max":       self._ds.al_max,
            "input_cols":   self._ds.input_cols,
        }, path)

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "MovementHRPredictor":
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        p = cls(window_size=ckpt["window_size"], hidden=ckpt["net_config"]["hidden"],
                device=device)
        p._net = _MovementHRNet(**ckpt["net_config"]).to(p.device)
        p._net.load_state_dict(ckpt["net_state"])
        # Restore norm params
        dummy = object.__new__(MinuteLevelDataset)
        dummy.hr_min = ckpt["hr_min"]
        dummy.hr_max = ckpt["hr_max"]
        dummy.al_min = ckpt["al_min"]
        dummy.al_max = ckpt["al_max"]
        dummy.input_cols = ckpt["input_cols"]
        dummy.n_input_features = len(ckpt["input_cols"])
        dummy._raw_joint = pd.DataFrame()
        dummy._X = np.empty((0,))
        dummy._y = np.empty((0,))
        dummy._ts = []
        p._ds = dummy
        return p


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Sleep multivariate LSTM autoencoder
# ──────────────────────────────────────────────────────────────────────────────

class _SleepAENet(nn.Module):
    """
    LSTM sequence autoencoder for multivariate sleep signals.

    Encoder compresses a ``window_size``-step, ``n_signals``-channel sequence
    into a latent vector; decoder reconstructs the original sequence.

    Input / output:  (batch, window_size, n_signals)
    """

    def __init__(
        self,
        n_signals: int,
        window_size: int,
        latent: int = 16,
        hidden: int = 32,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.n_signals = n_signals

        self.encoder = nn.LSTM(n_signals, hidden, num_layers=1, batch_first=True)
        self.enc_proj = nn.Linear(hidden, latent)

        self.latent_expand = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(hidden, hidden, num_layers=1, batch_first=True)
        self.dec_proj = nn.Linear(hidden, n_signals)

    def forward(self, x: torch.Tensor):
        """x : (B, T, C)  →  recon : (B, T, C)"""
        B, T, C = x.shape

        _, (h_enc, _) = self.encoder(x)          # h_enc: (1, B, H)
        z = self.enc_proj(h_enc.squeeze(0))       # (B, latent)

        # Repeat latent for each decoder step
        dec_in = self.latent_expand(z).unsqueeze(1).repeat(1, T, 1)  # (B, T, H)
        dec_out, _ = self.decoder(dec_in)         # (B, T, H)
        recon = torch.sigmoid(self.dec_proj(dec_out))  # (B, T, C) in [0,1]
        return recon


class SleepMultivariateAE:
    """
    Train and apply a multivariate LSTM autoencoder on sleep signals.

    Parameters
    ----------
    window_size : int
        Minutes per sleep patch.
    latent : int
        Bottleneck dimension.
    hidden : int
        LSTM hidden size.
    device : str | None
        Auto-detected when None.
    """

    def __init__(
        self,
        window_size: int = 30,
        latent: int = 16,
        hidden: int = 32,
        device: str | None = None,
    ) -> None:
        self.window_size = window_size
        self.latent = latent
        self.hidden = hidden
        self.device = torch.device(_auto_device(device))
        self._net: _SleepAENet | None = None
        self._ds:  SleepWindowDataset | None = None
        self._threshold: float = 0.0  # anomaly threshold (95th percentile of train error)

    # ------------------------------------------------------------------

    def fit(
        self,
        ds: SleepWindowDataset,
        epochs: int = 80,
        batch_size: int = 128,
        val_split: float = 0.15,
        patience: int = 12,
        threshold_pct: float = 95.0,
        verbose: bool = True,
    ) -> list[float]:
        """
        Train the autoencoder and calibrate the anomaly threshold.

        The threshold is set to the ``threshold_pct``-th percentile of
        per-window mean reconstruction error on the training set, so that
        only genuinely unusual windows are flagged.

        Returns training loss history.
        """
        self._ds = ds
        n_val   = max(1, int(len(ds) * val_split))
        n_train = len(ds) - n_val
        train_ds = Subset(ds, list(range(n_train)))
        val_ds   = Subset(ds, list(range(n_train, len(ds))))

        self._net = _SleepAENet(
            n_signals=ds.n_signals,
            window_size=self.window_size,
            latent=self.latent,
            hidden=self.hidden,
        ).to(self.device)

        optim = torch.optim.Adam(self._net.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, patience=6, factor=0.5, min_lr=1e-5
        )
        crit  = nn.MSELoss(reduction="mean")
        best_v, best_w, no_imp = float("inf"), None, 0
        history: list[float] = []

        t_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
        v_loader = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

        for epoch in range(1, epochs + 1):
            self._net.train()
            t_losses = []
            for x_b, y_b in t_loader:
                x_b = x_b.to(self.device)
                optim.zero_grad()
                recon = self._net(x_b)
                loss  = crit(recon, x_b)
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                optim.step()
                t_losses.append(loss.item())

            self._net.eval()
            v_losses = []
            with torch.no_grad():
                for x_b, _ in v_loader:
                    x_b = x_b.to(self.device)
                    v_losses.append(crit(self._net(x_b), x_b).item())

            t_l = float(np.mean(t_losses))
            v_l = float(np.mean(v_losses)) if v_losses else 0.0
            sched.step(v_l)
            history.append(t_l)

            if v_l < best_v:
                best_v = v_l
                best_w = {k: v.cpu().clone() for k, v in self._net.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1

            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"  [Sleep AE] epoch {epoch:3d} | train {t_l:.5f} | val {v_l:.5f}")

            if no_imp >= patience:
                if verbose:
                    print(f"  Early stop at epoch {epoch}.")
                break

        if best_w is not None:
            self._net.load_state_dict(best_w)

        # ── Calibrate threshold on training set reconstruction errors ──
        self._threshold = self._calibrate_threshold(train_ds, threshold_pct)
        if verbose:
            print(f"  Anomaly threshold (p{threshold_pct:.0f}): {self._threshold:.5f}")

        return history

    def _calibrate_threshold(self, ds_subset, pct: float) -> float:
        self._net.eval()
        loader = DataLoader(ds_subset, batch_size=256, shuffle=False)
        all_errors: list[float] = []
        with torch.no_grad():
            for x_b, _ in loader:
                x_b = x_b.to(self.device)
                recon = self._net(x_b)
                err = ((recon - x_b) ** 2).mean(dim=(1, 2)).cpu().numpy()
                all_errors.extend(err.tolist())
        return float(np.percentile(all_errors, pct))

    # ------------------------------------------------------------------

    def anomaly_series(self) -> pd.DataFrame:
        """
        Compute per-minute and per-night anomaly scores over all sleep data.

        Returns
        -------
        pd.DataFrame indexed by UTC timestamp with columns:
            recon_error         — per-minute mean squared reconstruction error
            anomaly_score       — z-score of recon_error within each night
            anomaly_flag        — True where recon_error > calibrated threshold
            night_date          — calendar date of the sleep session
            signal_errors       — dict of per-signal reconstruction errors
                                  (available as separate columns: hr_error,
                                   activity_error, rr_error)
        """
        if self._net is None or self._ds is None:
            raise RuntimeError("Call fit() before anomaly_series().")

        ds = self._ds
        loader = DataLoader(ds, batch_size=256, shuffle=False)

        self._net.eval()
        all_recon: list[np.ndarray] = []
        all_input: list[np.ndarray] = []
        with torch.no_grad():
            for x_b, _ in loader:
                x_b = x_b.to(self.device)
                recon = self._net(x_b)
                all_recon.append(recon.cpu().numpy())
                all_input.append(x_b.cpu().numpy())

        recon_arr = np.concatenate(all_recon, axis=0)   # (N, W, C)
        input_arr = np.concatenate(all_input, axis=0)   # (N, W, C)
        err_arr   = (recon_arr - input_arr) ** 2        # (N, W, C)

        # Per-window, per-step, per-signal error: aggregate via overlapping windows
        accum_total: dict[pd.Timestamp, list[float]] = {}
        accum_sig:   dict[pd.Timestamp, dict[str, list[float]]] = {}
        accum_night: dict[pd.Timestamp, str] = {}

        for i, ts_arr in enumerate(ds._ts):
            night = ds._nights[i]
            for j, ts in enumerate(ts_arr):
                t = pd.Timestamp(ts)
                accum_total.setdefault(t, []).append(float(err_arr[i, j].mean()))
                sig_d = accum_sig.setdefault(t, {c: [] for c in ds.signal_cols})
                for k, col in enumerate(ds.signal_cols):
                    sig_d[col].append(float(err_arr[i, j, k]))
                accum_night[t] = night

        ts_sorted = sorted(accum_total.keys())
        total_err = np.array([np.mean(accum_total[t]) for t in ts_sorted])
        nights    = [accum_night[t] for t in ts_sorted]

        result = pd.DataFrame({"recon_error": total_err, "night_date": nights},
                               index=ts_sorted)

        # Per-signal error columns
        for col in ds.signal_cols:
            key = col.replace("heart_rate", "hr").replace("activity_level", "activity") \
                     .replace("respiratory_rate", "rr") + "_error"
            result[key] = [np.mean(accum_sig[t][col]) for t in ts_sorted]

        # Anomaly flag from calibrated threshold
        result["anomaly_flag"] = result["recon_error"] > self._threshold

        # Per-night z-score (makes inter-night comparison easier)
        z_scores: list[float] = []
        for night, grp in result.groupby("night_date"):
            mu  = grp["recon_error"].mean()
            sig = grp["recon_error"].std()
            if sig == 0 or pd.isna(sig):
                sig = 1e-6
            z_scores.extend(((grp["recon_error"] - mu) / sig).tolist())
        result["anomaly_score"] = z_scores

        return result

    def night_summary(self) -> pd.DataFrame:
        """
        Per-night summary: mean reconstruction error + overall anomaly label.

        Returns
        -------
        pd.DataFrame indexed by night_date with columns:
            mean_recon_error, max_recon_error,
            anomaly_minutes, anomaly_pct,
            overall_anomaly  (True if >10% of minutes flagged)
        """
        minute_df = self.anomaly_series()
        rows: list[dict] = []
        for night, grp in minute_df.groupby("night_date"):
            total_min = len(grp)
            anom_min  = int(grp["anomaly_flag"].sum())
            rows.append({
                "night_date":        night,
                "mean_recon_error":  round(float(grp["recon_error"].mean()), 6),
                "max_recon_error":   round(float(grp["recon_error"].max()),  6),
                "anomaly_minutes":   anom_min,
                "anomaly_pct":       round(anom_min / max(total_min, 1) * 100, 1),
                "overall_anomaly":   anom_min / max(total_min, 1) > 0.10,
            })
        return (
            pd.DataFrame(rows)
            .set_index("night_date")
            .sort_index()
        )

    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        torch.save({
            "net_state":   self._net.state_dict(),
            "net_config":  {
                "n_signals":   self._ds.n_signals,
                "window_size": self.window_size,
                "latent":      self.latent,
                "hidden":      self.hidden,
            },
            "signal_cols": self._ds.signal_cols,
            "norm_params": self._ds.norm_params,
            "threshold":   self._threshold,
        }, path)

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "SleepMultivariateAE":
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        cfg  = ckpt["net_config"]
        ae   = cls(window_size=cfg["window_size"], latent=cfg["latent"],
                   hidden=cfg["hidden"], device=device)
        ae._net = _SleepAENet(**cfg).to(ae.device)
        ae._net.load_state_dict(ckpt["net_state"])
        ae._threshold = ckpt["threshold"]
        return ae


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

class SignalAnomalyEngine:
    """
    High-level orchestrator: build datasets, train both models, return anomaly
    DataFrames ready for analysis or plotting.

    Parameters
    ----------
    engine : HealthFeatureEngine
        Loaded feature engine.
    device : str | None
        'cuda', 'mps', or 'cpu' (auto-detected when None).

    Example
    -------
    ::

        sae = SignalAnomalyEngine(health_feature_engine)
        sae.fit_all()

        waking = sae.analyse_waking()
        print(waking[waking["anomaly_flag"]].head(20))

        sleep  = sae.analyse_sleep()
        nightly = sae.sleep_model.night_summary()
        print(nightly[nightly["overall_anomaly"]])
    """

    def __init__(self, engine, device: str | None = None) -> None:
        self.engine = engine
        self.device = device
        self.waking_model: MovementHRPredictor | None = None
        self.sleep_model:  SleepMultivariateAE  | None = None
        self._waking_ds:   MinuteLevelDataset   | None = None
        self._sleep_ds:    SleepWindowDataset    | None = None

    def fit_all(
        self,
        waking_epochs: int = 60,
        sleep_epochs:  int = 80,
        verbose: bool = True,
    ) -> None:
        """Build datasets and train both models."""
        if verbose:
            print("=" * 55)
            print("  Step 1 / 2 — Waking HR↔Movement Predictor")
            print("=" * 55)
        self._fit_waking(epochs=waking_epochs, verbose=verbose)

        if verbose:
            print("\n" + "=" * 55)
            print("  Step 2 / 2 — Sleep Multivariate Autoencoder")
            print("=" * 55)
        self._fit_sleep(epochs=sleep_epochs, verbose=verbose)

    def _fit_waking(self, epochs: int, verbose: bool) -> None:
        if verbose:
            print("  Building minute-level HR + movement dataset …")
        self._waking_ds = MinuteLevelDataset(
            self.engine, window_size=15, stride=5, exclude_sleep=True
        )
        if verbose:
            print(f"  {len(self._waking_ds):,} training windows")
        self.waking_model = MovementHRPredictor(window_size=15, device=self.device)
        self.waking_model.fit(self._waking_ds, epochs=epochs, verbose=verbose)

    def _fit_sleep(self, epochs: int, verbose: bool) -> None:
        if verbose:
            print("  Building sleep signal dataset …")
        self._sleep_ds = SleepWindowDataset(
            self.engine, window_size=30, stride=5
        )
        n_nights = len(self._sleep_ds.metadata["night_date"].unique())
        n_sigs   = self._sleep_ds.n_signals
        if verbose:
            print(f"  {len(self._sleep_ds):,} patches across {n_nights} nights, "
                  f"{n_sigs} signals: {self._sleep_ds.signal_cols}")
        self.sleep_model = SleepMultivariateAE(
            window_size=30, device=self.device
        )
        self.sleep_model._ds = self._sleep_ds
        self.sleep_model.fit(self._sleep_ds, epochs=epochs, verbose=verbose)

    # ------------------------------------------------------------------

    def analyse_waking(self, z_threshold: float = 2.0) -> pd.DataFrame:
        """
        Return per-minute waking anomaly DataFrame.

        Adds human-readable ``interpretation`` column explaining what each
        anomaly type likely means.
        """
        if self.waking_model is None:
            raise RuntimeError("Call fit_all() first.")
        df = self.waking_model.anomaly_series(z_threshold=z_threshold)
        df["interpretation"] = df["anomaly_direction"].map({
            "elevated":  "HR elevated for given movement — possible stress, illness, or arousal",
            "depressed": "HR lower than expected for activity — improved fitness or bradycardia",
            "normal":    "",
        })
        return df

    def analyse_sleep(self) -> pd.DataFrame:
        """Return per-minute sleep anomaly DataFrame with signal-level breakdowns."""
        if self.sleep_model is None:
            raise RuntimeError("Call fit_all() first.")
        return self.sleep_model.anomaly_series()

    def print_waking_summary(self, df: pd.DataFrame | None = None) -> None:
        """Print a concise waking anomaly summary to stdout."""
        if df is None:
            df = self.analyse_waking()
        total      = len(df)
        n_elevated = int((df["anomaly_direction"] == "elevated").sum())
        n_depress  = int((df["anomaly_direction"] == "depressed").sum())
        pct_e = n_elevated / total * 100
        pct_d = n_depress  / total * 100
        print("\n═══ Waking HR Anomaly Summary ═══")
        print(f"  Total minutes analysed : {total:,}")
        print(f"  Elevated HR anomalies  : {n_elevated:,}  ({pct_e:.1f}%) — stress / illness candidates")
        print(f"  Depressed HR anomalies : {n_depress:,}  ({pct_d:.1f}%) — fitness / medication candidates")
        # Top anomaly days
        df_flag = df[df["anomaly_flag"]].copy()
        if not df_flag.empty:
            df_flag["date"] = pd.DatetimeIndex(df_flag.index).date
            top_days = (
                df_flag.groupby(["date", "anomaly_direction"])
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
                .head(5)
            )
            print("\n  Top anomalous days:")
            for _, row in top_days.iterrows():
                print(f"    {row['date']}  {row['anomaly_direction']:10s}  {row['count']:4d} min")

    def print_sleep_summary(self, df: pd.DataFrame | None = None) -> None:
        """Print a concise sleep anomaly summary to stdout."""
        if self.sleep_model is None:
            raise RuntimeError("Call fit_all() first.")
        nightly = self.sleep_model.night_summary()
        print("\n═══ Sleep Anomaly Summary ═══")
        print(f"  Nights analysed : {len(nightly)}")
        anom_nights = nightly[nightly["overall_anomaly"]]
        print(f"  Anomalous nights: {len(anom_nights)} "
              f"({len(anom_nights)/max(len(nightly),1)*100:.0f}%)")
        if not anom_nights.empty:
            print("\n  Most anomalous nights (>10% of minutes flagged):")
            for date, row in anom_nights.sort_values("anomaly_pct", ascending=False).head(8).iterrows():
                print(f"    {date}  {row['anomaly_pct']:5.1f}% flagged  "
                      f"(mean err {row['mean_recon_error']:.5f})")

    # ------------------------------------------------------------------

    def save(self, prefix: str = "health_signal") -> None:
        """Save both models with a common filename prefix."""
        if self.waking_model is not None:
            self.waking_model.save(f"{prefix}_waking.pt")
        if self.sleep_model is not None:
            self.sleep_model.save(f"{prefix}_sleep.pt")
        print(f"Models saved with prefix '{prefix}'.")

    @classmethod
    def load(
        cls,
        engine,
        prefix: str = "health_signal",
        device: str | None = None,
    ) -> "SignalAnomalyEngine":
        """Load previously saved models."""
        sae = cls(engine, device=device)
        p = Path(prefix)
        waking_path = Path(f"{prefix}_waking.pt")
        sleep_path  = Path(f"{prefix}_sleep.pt")
        if waking_path.exists():
            sae.waking_model = MovementHRPredictor.load(waking_path, device=device)
        if sleep_path.exists():
            sae.sleep_model = SleepMultivariateAE.load(sleep_path, device=device)
        return sae


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────

def _auto_device(device: str | None) -> str:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

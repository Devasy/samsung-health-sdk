"""
HealthModelTrainer — training loop, evaluation, and persistence for
HealthLSTMAttention.

Usage::

    from samsung_health_sdk import SamsungHealthParser
    from samsung_health_sdk.features import HealthFeatureEngine
    from samsung_health_sdk.ml import build_daily_features, HealthModelTrainer

    p   = SamsungHealthParser("path/to/export")
    eng = HealthFeatureEngine(p)
    df  = build_daily_features(eng)

    trainer = HealthModelTrainer(df, seq_len=14)
    history = trainer.fit(epochs=200)          # early-stops around epoch 80–120
    trainer.save("health_model.pt")

    # Later sessions:
    trainer2 = HealthModelTrainer.load("health_model.pt")
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from samsung_health_sdk.ml.dataset import HealthWindowDataset
from samsung_health_sdk.ml.model import HealthLSTMAttention

# Maps model output dict keys → target column order in HealthWindowDataset
_PRED_KEYS = ["sleep_quality", "hrv_readiness", "energy_index"]
_TGT_COLS  = ["sleep_quality_score", "hrv_readiness_score", "energy_index"]


class HealthModelTrainer:
    """
    Train and evaluate a HealthLSTMAttention model on daily health features.

    The trainer holds a reference to the full dataset, splits it chronologically
    (last ``val_split`` fraction as validation — no shuffling to preserve temporal
    order), and runs training with ReduceLROnPlateau + early stopping.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``build_daily_features()``.
    seq_len : int
        Input look-back window in days.
    hidden : int
        LSTM hidden units per direction.
    n_layers : int
        Stacked BiLSTM layers.
    dropout : float
        Dropout rate (applied between LSTM layers and in the shared trunk).
    lr : float
        Initial Adam learning rate.
    val_split : float
        Fraction of the dataset reserved for validation (chronological tail).
    device : str | None
        'cuda', 'mps', or 'cpu'.  Auto-detected when None.
    """

    def __init__(
        self,
        df,
        seq_len: int = 14,
        hidden: int = 64,
        n_layers: int = 2,
        dropout: float = 0.3,
        lr: float = 1e-3,
        val_split: float = 0.2,
        device: str | None = None,
    ) -> None:
        self.df = df
        self.seq_len = seq_len

        # Build dataset
        full_ds = HealthWindowDataset(df, seq_len=seq_len, augment=False)
        n_total = len(full_ds)
        n_val   = max(1, int(n_total * val_split))
        n_train = n_total - n_val

        # Chronological split (no shuffle — temporal order must be preserved)
        train_indices = list(range(n_train))
        val_indices   = list(range(n_train, n_total))
        self.train_ds = Subset(full_ds, train_indices)
        self.val_ds   = Subset(full_ds, val_indices)
        self.full_ds  = full_ds

        self.feature_cols = full_ds.feature_cols
        self.target_cols  = full_ds.target_cols
        n_features = len(self.feature_cols)

        # Device selection
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        # Model
        self.model = HealthLSTMAttention(
            n_features=n_features,
            seq_len=seq_len,
            hidden=hidden,
            n_layers=n_layers,
            dropout=dropout,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=15, factor=0.5, min_lr=1e-5
        )
        self.criterion = nn.HuberLoss(delta=0.1)  # robust to outlier nights

        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        self._best_val_loss = float("inf")
        self._best_state: dict[str, torch.Tensor] | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        epochs: int = 200,
        batch_size: int = 16,
        patience: int = 40,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """
        Train with early stopping.

        Parameters
        ----------
        epochs : int
            Maximum number of training epochs.
        batch_size : int
            Mini-batch size.  For datasets < 100 samples, 8–16 works well.
        patience : int
            Stop training if validation loss does not improve for this many
            consecutive epochs.
        verbose : bool
            Print epoch summaries every 20 epochs.

        Returns
        -------
        dict
            ``{"train_loss": [...], "val_loss": [...]}`` per epoch.
        """
        train_loader = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True,
                                  drop_last=False)
        val_loader   = DataLoader(self.val_ds,   batch_size=batch_size, shuffle=False)

        no_improve = 0

        for epoch in range(1, epochs + 1):
            # ── train ──
            self.model.train()
            t_losses: list[float] = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                preds, _ = self.model(x_batch)
                pred_t = self._stack_preds(preds, x_batch.size(0)) / 100.0
                loss = self.criterion(pred_t, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                t_losses.append(loss.item())

            # ── validate ──
            self.model.eval()
            v_losses: list[float] = []
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    preds, _ = self.model(x_batch)
                    pred_t = self._stack_preds(preds, x_batch.size(0)) / 100.0
                    v_losses.append(self.criterion(pred_t, y_batch).item())

            t_loss = float(np.mean(t_losses)) if t_losses else 0.0
            v_loss = float(np.mean(v_losses)) if v_losses else 0.0
            self.history["train_loss"].append(t_loss)
            self.history["val_loss"].append(v_loss)
            self.scheduler.step(v_loss)

            if v_loss < self._best_val_loss:
                self._best_val_loss = v_loss
                self._best_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                no_improve = 0
            else:
                no_improve += 1

            if verbose and (epoch % 20 == 0 or epoch == 1):
                lr_now = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch:4d} │ train {t_loss:.4f} │ val {v_loss:.4f} │ "
                    f"lr {lr_now:.6f}"
                )

            if no_improve >= patience:
                if verbose:
                    print(
                        f"\nEarly stop at epoch {epoch} "
                        f"(no val improvement for {patience} epochs)."
                    )
                break

        # Restore best checkpoint
        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
            if verbose:
                print(f"Best val loss: {self._best_val_loss:.4f}")

        return self.history

    def _stack_preds(self, preds: dict, batch_size: int) -> torch.Tensor:
        """Stack model output dict into (B, 3) tensor matching target column order."""
        cols = [preds.get(k, torch.zeros(batch_size, device=self.device))
                for k in _PRED_KEYS]
        return torch.stack(cols, dim=1)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Save model weights, architecture config, and normalisation stats.

        The saved file contains everything needed to reconstruct the model
        and run inference without access to the original DataFrame.
        """
        path = Path(path)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "model_config": {
                    "n_features": self.model.n_features,
                    "seq_len":    self.model.seq_len,
                    "hidden":     self.model.hidden,
                    "n_layers":   self.model.lstm.num_layers,
                    "dropout":    self.model.lstm.dropout,
                },
                "feature_cols": self.feature_cols,
                "target_cols":  self.target_cols,
                "feat_min":     self.full_ds.feat_min.to_dict(),
                "feat_max":     self.full_ds.feat_max.to_dict(),
                "tgt_min":      self.full_ds.tgt_min.to_dict(),
                "tgt_max":      self.full_ds.tgt_max.to_dict(),
                "history":      self.history,
                "best_val_loss": self._best_val_loss,
            },
            path,
        )
        print(f"Model saved → {path}  (val loss {self._best_val_loss:.4f})")

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "HealthModelTrainer":
        """
        Load a previously saved trainer checkpoint.

        The returned object has ``model``, ``feature_cols``, ``target_cols``,
        and ``history`` populated.  It does *not* have a live dataset —
        call ``InsightEngine`` directly for inference.

        Parameters
        ----------
        path : str | Path
            Path to a .pt file produced by ``HealthModelTrainer.save()``.
        device : str | None
            Target device.  Auto-detected when None.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        cfg  = ckpt["model_config"]

        model = HealthLSTMAttention(**cfg)
        model.load_state_dict(ckpt["model_state"])

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        model = model.to(torch.device(device))

        # Build a minimal trainer shell (no dataset)
        trainer = object.__new__(cls)
        trainer.model        = model
        trainer.device       = torch.device(device)
        trainer.feature_cols = ckpt["feature_cols"]
        trainer.target_cols  = ckpt["target_cols"]
        trainer.history      = ckpt.get("history", {})
        trainer._best_val_loss = ckpt.get("best_val_loss", float("inf"))
        trainer._best_state    = None
        trainer.df             = None
        trainer.seq_len        = cfg["seq_len"]
        # Reconstruct normalisation Series so InsightEngine can use them
        import pandas as pd
        trainer._feat_min = pd.Series(ckpt["feat_min"])
        trainer._feat_max = pd.Series(ckpt["feat_max"])
        return trainer

    # ------------------------------------------------------------------
    # Quick evaluation summary
    # ------------------------------------------------------------------

    def evaluate(self) -> dict[str, Any]:
        """
        Compute per-target MAE and RMSE on the validation set.

        Returns a dict with keys:  sleep_quality, hrv_readiness, energy_index
        each containing {"mae": float, "rmse": float}.
        """
        if self.val_ds is None or len(self.val_ds) == 0:
            return {}

        self.model.eval()
        all_pred: list[np.ndarray] = []
        all_true: list[np.ndarray] = []

        val_loader = DataLoader(self.val_ds, batch_size=32, shuffle=False)
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(self.device)
                preds, _ = self.model(x_batch)
                pred_t = self._stack_preds(preds, x_batch.size(0)).cpu().numpy() / 100.0
                all_pred.append(pred_t)
                all_true.append(y_batch.numpy())

        pred_arr = np.concatenate(all_pred, axis=0)
        true_arr = np.concatenate(all_true, axis=0)

        results: dict[str, Any] = {}
        tgt_range = self.full_ds.tgt_max - self.full_ds.tgt_min

        for i, (key, col) in enumerate(zip(_PRED_KEYS, self.target_cols)):
            # Predictions and true values are in normalised [0,1] space
            # — multiply by original range to get MAE in original units
            scale = float(tgt_range.get(col, 1.0))
            mae  = float(np.mean(np.abs(pred_arr[:, i] - true_arr[:, i])) * scale)
            rmse = float(np.sqrt(np.mean((pred_arr[:, i] - true_arr[:, i])**2)) * scale)
            results[key] = {"mae": round(mae, 2), "rmse": round(rmse, 2)}

        return results

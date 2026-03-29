"""
HealthLSTMAttention — Bidirectional LSTM with temporal self-attention
for multi-task health metric forecasting.

Architecture
------------
Input:  (batch, seq_len, n_features)
  │
  ├─ Bidirectional LSTM  (n_layers=2, hidden=64, dropout=0.3)
  │     → (batch, seq_len, 128)   [128 = 2 × hidden for both directions]
  │
  ├─ Temporal Attention (single-head soft-max over seq_len)
  │     → context vector  (batch, 128)
  │     → attention weights (batch, seq_len)   ← interpretable!
  │
  ├─ Shared trunk  Linear(128→64) → ReLU → Dropout → Linear(64→32) → ReLU
  │     → (batch, 32)
  │
  └─ Three independent task heads (one per prediction target)
       head_sleep   Linear(32→1) + Sigmoid × 100 → sleep_quality   [0,100]
       head_hrv     Linear(32→1) + Sigmoid × 100 → hrv_readiness   [0,100]
       head_energy  Linear(32→1) + Sigmoid × 100 → energy_index    [0,100]

Interpretability
----------------
The attention weights reveal *which past days* most influenced the
predictions.  A high weight on day t−k means patterns from k days ago
drove today's forecast.

Combined with gradient × input saliency (computed in InsightEngine),
you can also identify *which features* on *which days* mattered most.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _TemporalAttention(nn.Module):
    """Additive attention (Bahdanau-style) over the time dimension."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False),
        )

    def forward(self, lstm_out: torch.Tensor):
        """
        Parameters
        ----------
        lstm_out : Tensor (B, T, H)

        Returns
        -------
        context : Tensor (B, H)        – weighted sum of lstm_out
        weights : Tensor (B, T)        – attention weights (sum to 1 along T)
        """
        scores = self.score(lstm_out).squeeze(-1)  # (B, T)
        weights = F.softmax(scores, dim=-1)  # (B, T)
        context = (weights.unsqueeze(-1) * lstm_out).sum(dim=1)  # (B, H)
        return context, weights


class HealthLSTMAttention(nn.Module):
    """
    Multi-task Bidirectional LSTM + Attention for health metric forecasting.

    Parameters
    ----------
    n_features : int
        Number of input features per time step.
    seq_len : int
        Input sequence length (look-back window in days).
    hidden : int
        LSTM hidden state size per direction. Total LSTM output = 2 × hidden.
    n_layers : int
        Number of stacked BiLSTM layers.
    dropout : float
        Dropout applied between LSTM layers and in the trunk.
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int = 14,
        hidden: int = 64,
        n_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.hidden = hidden

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        lstm_out_dim = hidden * 2  # bidirectional

        self.attn = _TemporalAttention(lstm_out_dim)

        self.trunk = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.head_sleep = nn.Linear(32, 1)
        self.head_hrv = nn.Linear(32, 1)
        self.head_energy = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : Tensor (batch, seq_len, n_features)

        Returns
        -------
        preds : dict[str, Tensor]
            Keys: "sleep_quality", "hrv_readiness", "energy_index"
            Each value: Tensor (batch,) with scores in [0, 100].
        attn_weights : Tensor (batch, seq_len)
            Attention distribution over the input time steps.
            Higher weight on time step t means day t was more influential.
        """
        lstm_out, _ = self.lstm(x)  # (B, T, 2H)
        context, attn_w = self.attn(lstm_out)  # (B, 2H), (B, T)
        features = self.trunk(context)  # (B, 32)

        sleep = torch.sigmoid(self.head_sleep(features)).squeeze(-1) * 100
        hrv = torch.sigmoid(self.head_hrv(features)).squeeze(-1) * 100
        energy = torch.sigmoid(self.head_energy(features)).squeeze(-1) * 100

        return {
            "sleep_quality": sleep,
            "hrv_readiness": hrv,
            "energy_index": energy,
        }, attn_w

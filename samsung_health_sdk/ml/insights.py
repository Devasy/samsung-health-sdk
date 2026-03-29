"""
InsightEngine — generate actionable, plain-language insights from a
trained HealthLSTMAttention model.

Two main capabilities
---------------------
1. **predict_tomorrow()**
   Runs the model on the most recent ``seq_len`` days and returns:
   - Predicted scores for sleep quality, HRV readiness, and energy index
   - Temporal attention weights (which past days drove the forecast)
   - Gradient-based feature importance (which features mattered most)
   - A 3-line plain-language summary

2. **discover_correlations() / print_correlations()**
   Computes Spearman correlations between today's lifestyle factors
   (activity, stress, cardiac load …) and next-day health outcomes
   (sleep quality, HRV readiness).  Surfaces your most personally
   significant behaviour → health relationships in plain English.

Usage::

    from samsung_health_sdk import SamsungHealthParser
    from samsung_health_sdk.features import HealthFeatureEngine
    from samsung_health_sdk.ml import (
        build_daily_features, HealthModelTrainer, InsightEngine
    )

    p   = SamsungHealthParser("path/to/export")
    eng = HealthFeatureEngine(p)
    df  = build_daily_features(eng)

    trainer = HealthModelTrainer.load("health_model.pt")
    ie = InsightEngine(trainer.model, df, trainer.feature_cols)

    report = ie.predict_tomorrow()
    print(report["summary"])
    ie.print_correlations()
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from samsung_health_sdk.ml.dataset import HealthWindowDataset

# Human-readable labels for each feature column
_LABELS: dict[str, str] = {
    "sleep_quality_score":  "sleep quality",
    "efficiency_pct":       "sleep efficiency",
    "deep_min":             "deep sleep duration",
    "rem_min":              "REM sleep duration",
    "sleep_light_min":      "light sleep duration",
    "awake_min":            "time awake in bed",
    "total_h":              "total sleep hours",
    "fragmentation_index":  "sleep fragmentation",
    "rmssd_mean":           "HRV (RMSSD)",
    "hrv_readiness_score":  "HRV readiness",
    "hrv_deviation_pct":    "HRV deviation from baseline",
    "mean_stress":          "average stress level",
    "stress_deviation_pct": "stress elevation above baseline",
    "sedentary_min":        "sedentary time",
    "light_activity_min":   "light activity",
    "low_mod_min":          "low-moderate activity",
    "moderate_min":         "moderate activity",
    "vigorous_min":         "vigorous exercise",
    "active_min":           "total active minutes",
    "mean_hr_active":       "heart rate during activity",
    "rr_mean":              "respiratory rate",
    "restlessness_score":   "sleep restlessness",
    "cardiac_load":         "cardiac load (aerobic fitness proxy)",
    "energy_index":         "energy index",
}


class InsightEngine:
    """
    Generate predictions and personalised insights from a trained model.

    Parameters
    ----------
    model : HealthLSTMAttention
        A trained (or loaded) model instance.
    df : pd.DataFrame
        Full daily feature DataFrame (output of ``build_daily_features()``).
    feature_cols : list[str]
        Feature columns the model was trained on.
    seq_len : int
        Input window length in days (must match training).
    """

    def __init__(
        self,
        model,
        df: pd.DataFrame,
        feature_cols: list[str],
        seq_len: int = 14,
    ) -> None:
        self.model        = model
        self.df           = df
        self.feature_cols = feature_cols
        self.seq_len      = seq_len
        self.device       = next(model.parameters()).device

        # Build normalised dataset for inference (no augmentation)
        self._ds = HealthWindowDataset(
            df, seq_len=seq_len, feature_cols=feature_cols, augment=False
        )

    # ------------------------------------------------------------------
    # Tomorrow's prediction
    # ------------------------------------------------------------------

    def predict_tomorrow(self) -> dict[str, Any]:
        """
        Predict sleep quality, HRV readiness, and energy index for tomorrow.

        Returns
        -------
        dict with keys:

        ``predictions``
            ``{"sleep_quality": float, "hrv_readiness": float, "energy_index": float}``
            Scores in original 0–100 scale.

        ``attention``
            ``{date_str: weight}`` — attention weight assigned to each past day.
            The most recent day usually has the highest weight.

        ``top_drivers``
            List of 3 plain-language feature names most influential for the
            prediction (ranked by gradient × input saliency).

        ``feature_importance``
            Full dict mapping feature column → importance score.

        ``summary``
            3-line human-readable forecast paragraph.
        """
        if len(self._ds) == 0:
            return {
                "error": (
                    f"Not enough data for prediction — need at least "
                    f"{self.seq_len + 1} days, have {len(self.df)}."
                )
            }

        last_idx = len(self._ds) - 1
        x_raw, date_labels = self._ds.get_window(last_idx)

        self.model.eval()
        x_in = x_raw.unsqueeze(0).to(self.device)  # (1, T, F)
        x_in.requires_grad_(True)

        # Forward with gradients to compute feature saliency
        preds, attn_w = self.model(x_in)
        # Combine all heads into a scalar loss for saliency propagation
        total = sum(v for v in preds.values())
        total.backward()

        # Gradient × input importance: (F,) averaged over time steps
        grad_x_input = (x_in.grad * x_in).abs().mean(dim=1).squeeze(0)
        importance_arr = grad_x_input.detach().cpu().numpy()

        with torch.no_grad():
            preds, attn_w = self.model(x_in.detach())

        pred_sleep  = float(preds["sleep_quality"].item())
        pred_hrv    = float(preds["hrv_readiness"].item())
        pred_energy = float(preds["energy_index"].item())

        # Attention weights per past day
        attn_np = attn_w.squeeze(0).cpu().numpy()
        attn_by_date = {
            str(date_labels[i]): round(float(attn_np[i]), 4)
            for i in range(len(date_labels))
        }

        # Feature importance dict and top-3 list
        feature_importance = dict(zip(self.feature_cols, importance_arr.tolist()))
        top3_cols  = sorted(feature_importance, key=feature_importance.__getitem__, reverse=True)[:3]
        top_drivers = [_LABELS.get(c, c) for c in top3_cols]

        # Personal 30-day averages as comparison baseline
        recent   = self.df[self.feature_cols].tail(30)
        avg_sleep = float(recent["sleep_quality_score"].median()) \
            if "sleep_quality_score" in recent else 65.0
        avg_hrv   = float(recent["hrv_readiness_score"].median()) \
            if "hrv_readiness_score" in recent else 65.0

        summary = _build_summary(
            pred_sleep, pred_hrv, pred_energy,
            avg_sleep, avg_hrv, top_drivers,
        )

        return {
            "predictions": {
                "sleep_quality": round(pred_sleep, 1),
                "hrv_readiness": round(pred_hrv, 1),
                "energy_index":  round(pred_energy, 1),
            },
            "attention": attn_by_date,
            "top_drivers": top_drivers,
            "feature_importance": {k: round(v, 5) for k, v in feature_importance.items()},
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # Correlation discovery
    # ------------------------------------------------------------------

    def discover_correlations(
        self,
        min_samples: int = 14,
        min_abs_corr: float = 0.15,
    ) -> pd.DataFrame:
        """
        Compute Spearman correlations between today's lifestyle inputs and
        next-day health outcomes.

        For each (feature, outcome) pair with enough data points, the function
        shifts the outcome by one day, computes the rank correlation, and
        generates a plain-language insight string.

        Parameters
        ----------
        min_samples : int
            Minimum number of paired observations required to report a correlation.
        min_abs_corr : float
            Minimum |r| to include in results.

        Returns
        -------
        pd.DataFrame with columns:
            feature, feature_label, outcome, correlation, direction, insight
        sorted by |correlation| descending.
        """
        df = self.df.copy()
        outcome_cols = [c for c in ["sleep_quality_score", "hrv_readiness_score", "energy_index"]
                        if c in df.columns]
        lifestyle_cols = [
            c for c in self.feature_cols
            if c not in outcome_cols
        ]

        rows: list[dict] = []
        for outcome in outcome_cols:
            y = df[outcome].shift(-1)  # next-day outcome
            for feat in lifestyle_cols:
                x = df[feat]
                valid = x.notna() & y.notna()
                if valid.sum() < min_samples:
                    continue
                corr = float(x[valid].corr(y[valid], method="spearman"))
                if abs(corr) < min_abs_corr:
                    continue
                rows.append({
                    "feature":       feat,
                    "feature_label": _LABELS.get(feat, feat),
                    "outcome":       outcome,
                    "outcome_label": _LABELS.get(outcome, outcome),
                    "correlation":   round(corr, 3),
                    "direction":     "positive" if corr > 0 else "negative",
                    "insight":       _corr_sentence(feat, outcome, corr),
                })

        if not rows:
            return pd.DataFrame()

        return (
            pd.DataFrame(rows)
            .assign(_abs=lambda d: d["correlation"].abs())
            .sort_values("_abs", ascending=False)
            .drop(columns="_abs")
            .reset_index(drop=True)
        )

    def print_correlations(self, top_n: int = 10) -> None:
        """Print the top personalised correlation insights to stdout."""
        corr_df = self.discover_correlations()
        if corr_df.empty:
            print("Not enough data for correlation analysis (need ≥14 paired days).")
            return

        print("\n" + "═" * 60)
        print("   YOUR PERSONAL HEALTH CORRELATIONS")
        print("═" * 60)
        shown = 0
        seen_pairs: set[tuple] = set()
        for _, row in corr_df.iterrows():
            pair = (row["feature"], row["outcome"])
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            bar_len = int(abs(row["correlation"]) * 20)
            bar_dir = "+" if row["direction"] == "positive" else "-"
            bar = f"[{bar_dir * bar_len}{' ' * (20 - bar_len)}]"
            print(f"\n  {row['insight']}")
            print(f"  r = {row['correlation']:+.3f}  {bar}")
            shown += 1
            if shown >= top_n:
                break
        print("\n" + "═" * 60 + "\n")

    # ------------------------------------------------------------------
    # Pattern summaries
    # ------------------------------------------------------------------

    def summarise_patterns(self) -> list[str]:
        """
        Return a list of discovered patterns as plain-language strings.

        These are threshold-based discoveries that go beyond simple
        correlations — e.g. "After 3+ consecutive high-stress days, your
        deep sleep drops by X%".
        """
        df = self.df.copy()
        patterns: list[str] = []

        # ── Pattern 1: consecutive high-stress nights and deep sleep ──
        if "mean_stress" in df and "deep_min" in df:
            high_stress = df["mean_stress"] > df["mean_stress"].quantile(0.75)
            # Rolling 3-day count of high-stress days
            consec = high_stress.rolling(3, min_periods=3).sum()
            after_streak = consec >= 3
            deep_after   = df.loc[after_streak.shift(1).fillna(False), "deep_min"]
            deep_other   = df.loc[~after_streak.shift(1).fillna(False), "deep_min"]
            if len(deep_after) >= 5 and deep_other.notna().sum() >= 5:
                drop_pct = (deep_other.mean() - deep_after.mean()) / (deep_other.mean() + 1e-9) * 100
                if drop_pct > 5:
                    patterns.append(
                        f"After 3+ consecutive high-stress days, your deep sleep drops "
                        f"by {drop_pct:.0f}% on average."
                    )

        # ── Pattern 2: vigorous exercise and next-day HRV ──
        if "vigorous_min" in df and "hrv_readiness_score" in df:
            has_vigorous   = df["vigorous_min"] > 20
            hrv_after_vig  = df.loc[has_vigorous.shift(1).fillna(False), "hrv_readiness_score"]
            hrv_after_rest = df.loc[~has_vigorous.shift(1).fillna(False), "hrv_readiness_score"]
            if hrv_after_vig.notna().sum() >= 5 and hrv_after_rest.notna().sum() >= 5:
                diff = hrv_after_vig.mean() - hrv_after_rest.mean()
                direction = "higher" if diff > 0 else "lower"
                if abs(diff) > 2:
                    patterns.append(
                        f"Days after 20+ minutes of vigorous exercise, your HRV readiness "
                        f"is {abs(diff):.0f} points {direction} on average."
                    )

        # ── Pattern 3: late-night stress and sleep fragmentation ──
        if "stress_deviation_pct" in df and "fragmentation_index" in df:
            high_dev = df["stress_deviation_pct"] > 20
            frag_hi  = df.loc[high_dev, "fragmentation_index"]
            frag_lo  = df.loc[~high_dev, "fragmentation_index"]
            if frag_hi.notna().sum() >= 5 and frag_lo.notna().sum() >= 5:
                diff = frag_hi.mean() - frag_lo.mean()
                if diff > 0.1:
                    patterns.append(
                        f"On high-stress nights (stress >20% above baseline), your sleep "
                        f"fragmentation is {diff:.1f}× higher — more wake-ups per hour."
                    )

        # ── Pattern 4: sedentary days and next-day energy ──
        if "sedentary_min" in df and "energy_index" in df:
            very_sed    = df["sedentary_min"] > df["sedentary_min"].quantile(0.75)
            energy_next = df["energy_index"].shift(-1)
            e_sed  = energy_next[very_sed]
            e_actv = energy_next[~very_sed]
            if e_sed.notna().sum() >= 5 and e_actv.notna().sum() >= 5:
                diff = e_actv.mean() - e_sed.mean()
                if diff > 3:
                    patterns.append(
                        f"Your energy index the next day is {diff:.0f} points higher "
                        f"following active days versus very sedentary ones."
                    )

        if not patterns:
            patterns.append(
                "Not enough data yet to surface strong patterns — keep wearing the watch!"
            )
        return patterns

    def print_patterns(self) -> None:
        """Print all discovered behavioural patterns."""
        print("\n" + "═" * 60)
        print("   PATTERNS DISCOVERED IN YOUR DATA")
        print("═" * 60)
        for i, p in enumerate(self.summarise_patterns(), 1):
            print(f"\n  {i}. {p}")
        print("\n" + "═" * 60 + "\n")


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _level(score: float) -> str:
    if score >= 80:
        return "HIGH"
    if score >= 60:
        return "MODERATE"
    return "LOW"


def _delta_phrase(value: float, baseline: float, metric: str) -> str:
    delta = value - baseline
    if delta > 5:
        return f"{metric} is forecast above your recent average (+{delta:.0f} pts)"
    if delta < -5:
        return f"{metric} may be below your recent average ({delta:.0f} pts)"
    return f"{metric} looks close to your recent average"


def _build_summary(
    pred_sleep: float,
    pred_hrv: float,
    pred_energy: float,
    avg_sleep: float,
    avg_hrv: float,
    top_drivers: list[str],
) -> str:
    driver_str = " and ".join(top_drivers[:2]) if len(top_drivers) >= 2 else top_drivers[0]
    lines = [
        (
            f"Tomorrow's forecast:  "
            f"Sleep {pred_sleep:.0f}/100 ({_level(pred_sleep)})  •  "
            f"HRV readiness {pred_hrv:.0f}/100 ({_level(pred_hrv)})  •  "
            f"Energy {pred_energy:.0f}/100 ({_level(pred_energy)})"
        ),
        (
            f"{_delta_phrase(pred_sleep, avg_sleep, 'Sleep quality').capitalize()}.  "
            f"{_delta_phrase(pred_hrv, avg_hrv, 'HRV readiness').capitalize()}."
        ),
        f"Key drivers: {driver_str}.",
    ]
    return "\n".join(lines)


def _corr_sentence(feature: str, outcome: str, corr: float) -> str:
    feat_label    = _LABELS.get(feature, feature)
    outcome_label = _LABELS.get(outcome, outcome)
    more_or_less  = "more" if corr > 0 else "less"
    higher_or_lower = "higher" if corr > 0 else "lower"
    strength = (
        "strongly" if abs(corr) > 0.5 else
        "moderately" if abs(corr) > 0.3 else
        "weakly"
    )
    return (
        f"Days with {more_or_less} {feat_label} are {strength} associated with "
        f"{higher_or_lower} {outcome_label} the next day."
    )

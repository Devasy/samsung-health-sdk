"""
samsung_health_sdk.ml
=====================
Deep-learning module for health metric forecasting and personalised insight generation.

Requires PyTorch::

    pip install samsung-health-sdk[ml]

Quick start::

    from samsung_health_sdk import SamsungHealthParser
    from samsung_health_sdk.features import HealthFeatureEngine
    from samsung_health_sdk.ml import build_daily_features, HealthModelTrainer, InsightEngine

    p   = SamsungHealthParser("path/to/export")
    eng = HealthFeatureEngine(p)
    df  = build_daily_features(eng)          # one row per day, ~23 features

    trainer = HealthModelTrainer(df)
    trainer.fit(epochs=200)                  # trains in <60 s on CPU
    trainer.save("health_model.pt")

    ie = InsightEngine(trainer.model, df, trainer.feature_cols)
    print(ie.predict_tomorrow()["summary"])  # plain-language forecast
    ie.print_correlations()                  # personalised correlation insights
"""

from samsung_health_sdk.ml.feature_matrix import build_daily_features, FEATURE_COLS, TARGET_COLS
from samsung_health_sdk.ml.dataset import HealthWindowDataset
from samsung_health_sdk.ml.model import HealthLSTMAttention
from samsung_health_sdk.ml.trainer import HealthModelTrainer
from samsung_health_sdk.ml.insights import InsightEngine
from samsung_health_sdk.ml.signal_dataset import MinuteLevelDataset, SleepWindowDataset
from samsung_health_sdk.ml.signal_models import (
    MovementHRPredictor,
    SleepMultivariateAE,
    SignalAnomalyEngine,
)

__all__ = [
    # Daily-level forecasting
    "build_daily_features",
    "FEATURE_COLS",
    "TARGET_COLS",
    "HealthWindowDataset",
    "HealthLSTMAttention",
    "HealthModelTrainer",
    "InsightEngine",
    # Minute-level anomaly detection
    "MinuteLevelDataset",
    "SleepWindowDataset",
    "MovementHRPredictor",
    "SleepMultivariateAE",
    "SignalAnomalyEngine",
]

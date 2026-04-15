"""
End-to-end example: train a DL model on Samsung Health data and get insights.

Install:
    pip install -e ".[ml]"

Run:
    python ml_example.py
"""
from samsung_health_sdk import SamsungHealthParser
from samsung_health_sdk.features import HealthFeatureEngine
from samsung_health_sdk.ml import (
    build_daily_features,
    HealthModelTrainer,
    InsightEngine,
)

# ── 1. Load your Samsung Health export ────────────────────────────────────────
EXPORT_DIR = "samsunghealth_patel.devasy.23_20250630001879"

p   = SamsungHealthParser(EXPORT_DIR)
eng = HealthFeatureEngine(p, tz_offset_hours=5.5)   # IST = UTC+5:30

# ── 2. Build the daily feature matrix (~23 columns, one row per day) ──────────
print("Building daily feature matrix …")
df = build_daily_features(eng)
print(f"  {len(df)} days  ×  {df.shape[1]} features")
print(df.tail(5).to_string())

# ── 3. Train the model ────────────────────────────────────────────────────────
print("\nTraining HealthLSTMAttention …")
trainer = HealthModelTrainer(
    df,
    seq_len=14,      # 14-day look-back window
    hidden=64,       # LSTM hidden units per direction
    n_layers=2,
    dropout=0.3,
    lr=1e-3,
    val_split=0.2,
)
history = trainer.fit(epochs=300, batch_size=16, patience=40)

# Validation metrics
print("\nValidation performance:")
metrics = trainer.evaluate()
for target, scores in metrics.items():
    print(f"  {target:20s}  MAE={scores['mae']:.1f}  RMSE={scores['rmse']:.1f}")

# Save for future sessions
trainer.save("health_model.pt")

# ── 4. Generate insights ──────────────────────────────────────────────────────
ie = InsightEngine(trainer.model, df, trainer.feature_cols, seq_len=14)

print("\n" + "=" * 60)
report = ie.predict_tomorrow()
if "error" in report:
    print(report["error"])
else:
    print(report["summary"])
    print("\nAttention (most influential past days):")
    sorted_attn = sorted(report["attention"].items(), key=lambda x: x[1], reverse=True)
    for date, weight in sorted_attn[:5]:
        print(f"  {date}  →  {weight:.3f}")

# Personalised correlations
ie.print_correlations(top_n=8)

# Behavioural patterns
ie.print_patterns()

# ── 5. (Optional) Load a saved model and run inference ────────────────────────
# trainer2 = HealthModelTrainer.load("health_model.pt")
# ie2 = InsightEngine(trainer2.model, df, trainer2.feature_cols)
# print(ie2.predict_tomorrow()["summary"])

"""
Minute-level cross-signal anomaly detection example.

Run:
    python anomaly_example.py
"""
from samsung_health_sdk import SamsungHealthParser
from samsung_health_sdk.features import HealthFeatureEngine
from samsung_health_sdk.ml import SignalAnomalyEngine

EXPORT_DIR = "samsunghealth_patel.devasy.23_20250630001879"

p   = SamsungHealthParser(EXPORT_DIR)
eng = HealthFeatureEngine(p, tz_offset_hours=5.5)

# ── Train both models (takes 1–3 min on CPU) ──────────────────────────────────
sae = SignalAnomalyEngine(eng)
sae.fit_all(waking_epochs=60, sleep_epochs=80)
sae.save("health_signal")   # → health_signal_waking.pt, health_signal_sleep.pt

# ── Waking: HR vs movement anomalies ──────────────────────────────────────────
waking_df = sae.analyse_waking(z_threshold=2.0)
sae.print_waking_summary(waking_df)

# Inspect specific anomalous windows
elevated = waking_df[waking_df["anomaly_direction"] == "elevated"]
print("\nSample elevated-HR anomaly windows:")
print(
    elevated[["actual_hr", "predicted_hr", "residual", "z_score"]]
    .head(10)
    .round(1)
    .to_string()
)

# ── Sleep: multi-signal autoencoder anomalies ──────────────────────────────────
sleep_df = sae.analyse_sleep()
sae.print_sleep_summary()

# Per-night summary
nightly = sae.sleep_model.night_summary()
print("\nAll nights:")
print(nightly.to_string())

# Zoom into most anomalous night
if not nightly[nightly["overall_anomaly"]].empty:
    worst_night = nightly[nightly["overall_anomaly"]]["anomaly_pct"].idxmax()
    night_mins  = sleep_df[sleep_df["night_date"] == worst_night]
    print(f"\nMinute-level breakdown for worst night ({worst_night}):")
    print(
        night_mins[["recon_error", "anomaly_flag", "anomaly_score"] +
                   [c for c in night_mins.columns if c.endswith("_error")]]
        .head(30)
        .round(4)
        .to_string()
    )

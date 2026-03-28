"""
Quick-start example for samsung-health-sdk.

Run from the SHA/ directory:
    python example.py
"""

import sys

# Force UTF-8 output on Windows
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from pathlib import Path
from samsung_health_sdk import SamsungHealthParser, SamsungHealthComparator

# Point at the export directory
EXPORT_DIR = Path(__file__).parent / "samsunghealth_patel.devasy.23_20250630001879"

p = SamsungHealthParser(EXPORT_DIR)

# --- Discovery ---
metrics = p.list_metrics()
print(f"\nAvailable metrics ({len(metrics)}):")
for m in metrics:
    print(f"  {m}")

# --- Date window ---
START = "2024-10-01"
END = "2024-10-31"

# --- Heart rate ---
print("\n[Heart Rate (summary)]")
hr = p.get_heart_rate(START, END)
print(f"Rows: {len(hr)}")
print(hr[["start_time", "heart_rate", "heart_rate_min", "heart_rate_max"]].head(5).to_string())

print("\n[Heart Rate (minute-level)]")
hr_min = p.get_heart_rate(START, END, granularity="minute")
print(f"Rows: {len(hr_min)}")
print(hr_min[["start_time", "heart_rate", "heart_rate_min", "heart_rate_max"]].head(5).to_string())

# --- Sleep ---
print("\n[Sleep Stages]")
sleep = p.get_sleep(START, END)
print(f"Rows: {len(sleep)}")
print(sleep[["start_time", "end_time", "stage_label", "sleep_id"]].head(8).to_string())
if not sleep.empty:
    print("\nStage distribution:")
    print(sleep["stage_label"].value_counts().to_string())

# --- Stress ---
print("\n[Stress]")
stress = p.get_stress(START, END)
print(f"Rows: {len(stress)}")
print(stress[["start_time", "score", "stress_min", "stress_max"]].head(5).to_string())

# --- Skin temperature ---
print("\n[Skin Temperature (summary)]")
skin = p.get_skin_temperature(START, END)
print(f"Rows: {len(skin)}")
if not skin.empty:
    print(
        skin[["start_time", "temperature", "temperature_min", "temperature_max"]]
        .head(5)
        .to_string()
    )

# --- Respiratory rate (November - when data exists) ---
print("\n[Respiratory Rate (minute-level, November 2024)]")
rr = p.get_respiratory_rate("2024-11-01", "2024-11-30", granularity="minute")
print(f"Rows: {len(rr)}")
print(rr[["start_time", "respiratory_rate"]].head(5).to_string())

# --- HRV ---
print("\n[HRV]")
hrv = p.get_hrv("2024-11-01", "2024-11-30")
print(f"Rows: {len(hrv)}")
if not hrv.empty:
    print(hrv.head(3).to_string())

# --- Exercise ---
print("\n[Exercise sessions]")
ex = p.get_exercise("2024-01-01", "2025-06-30")
print(f"Rows: {len(ex)}")
if not ex.empty:
    print(
        ex[
            [
                "start_time",
                "exercise_name",
                "duration_sec",
                "distance",
                "calorie",
                "mean_heart_rate",
            ]
        ]
        .head(5)
        .to_string()
    )

# --- Steps ---
print("\n[Steps (May-Jun 2025)]")
steps = p.get_steps("2025-05-01", "2025-06-30")
print(f"Rows: {len(steps)}")
print(steps[["start_time", "count", "walk_step", "distance"]].head(5).to_string())

# --- Generic accessor ---
print("\n[Generic accessor: vitality_score]")
try:
    vs = p.get_metric("com.samsung.shealth.vitality_score", start=START, end=END)
    print(f"Rows: {len(vs)}")
    print(vs.head(3).to_string())
except Exception as e:
    print(f"  (not available: {e})")

# --- Multi-person comparator (same person, demonstrates API) ---
print("\n[Comparator: compare heart rate for same person]")
comp = SamsungHealthComparator({"Oct": p, "Nov": p})
df_oct = comp.compare_heart_rate(start=START, end=END, persons=["Oct"])
print(f"Comparison rows: {len(df_oct)}")
print(df_oct[["person", "start_time", "heart_rate"]].head(5).to_string())

print("\n[Comparator: time_shift=True]")
df_shifted = comp.compare_heart_rate(start=START, end=END, persons=["Oct"], time_shift=True)
print(df_shifted[["person", "relative_time", "heart_rate"]].head(5).to_string())

print("\nDone.")

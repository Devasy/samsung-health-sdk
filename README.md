# samsung-health-sdk

A Python SDK for parsing and analysing [Samsung Health](https://www.samsung.com/global/galaxy/apps/samsung-health/) export data.

Load any health metric from a Samsung Health export directory as a **pandas DataFrame** with a single function call. Compare data across multiple people or time windows. Derive higher-level health features from the raw data.

## Installation

```bash
pip install samsung-health-sdk
```

## Quick Start

```python
from samsung_health_sdk import SamsungHealthParser, SamsungHealthComparator

# Point at your Samsung Health export directory
p = SamsungHealthParser("path/to/samsunghealth_export_dir")

# See all available metrics
print(p.list_metrics())

# Load heart rate (hourly summaries or minute-level)
hr     = p.get_heart_rate("2024-10-01", "2024-10-31")
hr_min = p.get_heart_rate("2024-10-01", "2024-10-31", granularity="minute")

# All supported metrics
sleep  = p.get_sleep("2024-10-01", "2024-10-31")
skin   = p.get_skin_temperature("2024-10-01", "2024-10-31", granularity="minute")
stress = p.get_stress("2024-10-01", "2024-10-31")
spo2   = p.get_spo2("2024-10-01", "2024-10-31")
steps  = p.get_steps("2024-10-01", "2024-10-31")
hrv    = p.get_hrv("2024-10-01", "2024-10-31")
rr     = p.get_respiratory_rate("2024-10-01", "2024-10-31", granularity="minute")
ex     = p.get_exercise("2024-10-01", "2024-10-31")
mv     = p.get_movement("2024-10-01", "2024-10-31")  # per-minute activity_level

# Generic accessor for any metric by its full name
df = p.get_metric("com.samsung.shealth.vitality_score", start="2024-10-01")
```

## Feature Engineering

`HealthFeatureEngine` derives meaningful higher-level features from the raw data:

```python
from samsung_health_sdk.features import HealthFeatureEngine

eng = HealthFeatureEngine(p, tz_offset_hours=5.5)  # tz_offset_hours: your UTC offset

# Per-night sleep quality: efficiency, deep/REM %, fragmentation, composite score
sleep_stats = eng.sleep_sessions("2025-01-01", "2025-03-31")

# Per-night HRV + respiratory rate + movement restlessness during sleep
physio = eng.nightly_physiology("2025-01-01", "2025-03-31")
# Columns: rmssd_mean, rmssd_min, rmssd_std, rr_mean, rr_std,
#          restlessness_score, restless_min, hrv_suppression_flag

# HRV readiness: today vs your rolling N-day personal baseline
readiness = eng.hrv_readiness("2025-01-01", "2025-03-31", baseline_days=14)
# Columns: rmssd_mean, baseline_14d, deviation_pct, readiness_score, low_readiness_flag

# Previous-day stress deviation vs that night's sleep quality
impact = eng.stress_impact_on_sleep("2025-01-01", "2025-03-31")
# Uses stress deviation from rolling baseline, not absolute score

# Per-day activity breakdown + HR context + stress
profile = eng.daily_activity_profile("2025-01-01", "2025-03-31")
# Columns: sedentary_min, light_min, low_mod_min, moderate_min, vigorous_min,
#          active_min, mean_hr_active, median_hr_day, mean_stress, stress_deviation_pct

# Walking cardiac load trend (HR / speed — lower = more aerobically efficient)
cardiac = eng.walking_cardiac_load("2024-11-01", "2025-06-30", source="auto")
# source='auto': pedometer (most accurate) > movement (accelerometer-based,
#                extends to Nov 2024) > exercise summaries (Jun 2022+)
# Columns: date, duration_min, distance_m, speed_mps, mean_hr, cardiac_load,
#          source, rolling_4w_cardiac_load, cardiac_load_trend
```

## Multi-Person Comparison

```python
p1 = SamsungHealthParser("path/to/person1_export")
p2 = SamsungHealthParser("path/to/person2_export")

comp = SamsungHealthComparator({"Alice": p1, "Bob": p2})

# Compare heart rate — absolute calendar window
df = comp.compare_heart_rate("2024-10-01", "2024-10-31")

# Align to relative Day 0 per person (time_shift=True)
df = comp.compare_heart_rate("2024-10-01", "2024-10-31", time_shift=True)
```

## Export Format

Samsung Health exports a directory containing:
- **CSV files** per health metric (some with a metadata row, some without — auto-detected)
- **`jsons/`** subdirectory with per-minute binning JSON files
- **`files/`** subdirectory with binary attachments (ECG waveforms, photos)

The SDK handles BOM encoding, namespaced column headers, UTC offset timestamps, trailing commas, and lazy JSON loading automatically.

## Sleep Stage Codes

| Code  | Label  |
|-------|--------|
| 40001 | Awake  |
| 40002 | Light  |
| 40003 | Deep   |
| 40004 | REM    |

## Movement Activity Levels

Per-minute accelerometer intensity from `get_movement()`:

| Range    | Intensity    |
|----------|-------------|
| 0–5      | Sedentary   |
| 5–20     | Light       |
| 20–50    | Low-moderate|
| 50–100   | Moderate    |
| 100+     | Vigorous    |

## Requirements

- Python 3.9+
- pandas >= 2.0
- numpy >= 1.24

## Contributing

```bash
git clone https://github.com/your-username/samsung-health-sdk
cd samsung-health-sdk
pip install -e ".[dev]"
pytest
```

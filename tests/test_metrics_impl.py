import pytest
import pandas as pd
from pathlib import Path

from samsung_health_sdk.metrics.heart_rate import HeartRateMetric
from samsung_health_sdk.metrics.sleep import SleepStageMetric

def test_heart_rate_load_summary(tmp_path):
    f = tmp_path / "com.samsung.shealth.tracker.heart_rate.123.csv"
    f.write_text("start_time,heart_rate,min,max\n2024-01-01 10:00:00,75,60,100")

    m = HeartRateMetric(tmp_path)
    df = m.load_summary()
    assert "heart_rate_min" in df.columns
    assert "heart_rate_max" in df.columns
    assert "min" not in df.columns
    assert "max" not in df.columns
    assert df.iloc[0]["heart_rate_min"] == 60

def test_heart_rate_load_detail(tmp_path):
    f = tmp_path / "com.samsung.shealth.tracker.heart_rate.123.csv"
    f.write_text("start_time,heart_rate,min,max\n2024-01-01 10:00:00,75,60,100")

    m = HeartRateMetric(tmp_path)
    df = m.load_detail()
    # If no binning, fallback should still have original columns based on select_columns,
    # but base metric load_detail calls select_columns(summary) and HeartRateMetric doesn't
    # override load_detail mapping except returning what base gives it.
    # Wait, base gives it select_columns(summary), which includes 'min' and 'max' since they are in value_columns.
    assert not df.empty

def test_sleep_load_summary(tmp_path):
    f = tmp_path / "com.samsung.health.sleep_stage.123.csv"
    f.write_text("start_time,stage\n2024-01-01 10:00:00,40001\n2024-01-01 10:05:00,40002")

    m = SleepStageMetric(tmp_path)
    df = m.load_summary()
    assert "stage_label" in df.columns
    assert df.iloc[0]["stage_label"] == "Awake"
    assert df.iloc[1]["stage_label"] == "Light"

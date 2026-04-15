import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
import json

from samsung_health_sdk.utils import (
    _offset_to_timedelta,
    _strip_namespace,
    _detect_skip_rows,
    read_csv,
    parse_timestamps,
    resolve_binning_path,
    load_binning_json,
    coerce_date,
    filter_date_range,
)

def test_offset_to_timedelta():
    assert _offset_to_timedelta("UTC+0530") == timedelta(hours=5, minutes=30)
    assert _offset_to_timedelta("UTC-0800") == timedelta(hours=-8, minutes=0)
    assert _offset_to_timedelta("UTC+0000") == timedelta(0)
    assert _offset_to_timedelta("invalid") == timedelta(0)
    assert _offset_to_timedelta("") == timedelta(0)

def test_strip_namespace():
    assert _strip_namespace("com.samsung.health.heart_rate.start_time") == "start_time"
    assert _strip_namespace("start_time") == "start_time"

def test_detect_skip_rows(tmp_path):
    f1 = tmp_path / "with_meta.csv"
    f1.write_text("com.samsung.health.heart_rate,123,1.0\nstart_time,heart_rate\n2024-01-01,75")
    assert _detect_skip_rows(f1) == 1

    f2 = tmp_path / "no_meta.csv"
    f2.write_text("start_time,heart_rate\n2024-01-01,75")
    assert _detect_skip_rows(f2) == 0

    f3 = tmp_path / "does_not_exist.csv"
    assert _detect_skip_rows(f3) == 0

def test_read_csv(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("com.samsung.health.heart_rate,123,1.0\nstart_time,com.samsung.health.heart_rate,time_offset\n2024-01-01 10:00:00,75,UTC+0000")
    df = read_csv(f)
    assert list(df.columns) == ["start_time", "heart_rate", "time_offset"]
    assert len(df) == 1
    assert df.iloc[0]["heart_rate"] == 75

def test_read_csv_parse_error(tmp_path):
    from samsung_health_sdk.exceptions import DataParseError
    f = tmp_path / "bad.csv"
    f.write_text("start_time\n\"unclosed quote")
    with pytest.raises(DataParseError):
        read_csv(f)

def test_parse_timestamps():
    df = pd.DataFrame({
        "start_time": ["2024-01-01 12:00:00.000"],
        "time_offset": ["UTC+0530"],
        "update_time": ["2024-01-01 12:00:00.000"]
    })
    parsed = parse_timestamps(df)
    # The naive time 12:00:00 +0530 means UTC is 06:30:00
    assert parsed.iloc[0]["start_time"] == pd.Timestamp("2024-01-01 06:30:00", tz="UTC")
    assert parsed.iloc[0]["update_time"] == pd.Timestamp("2024-01-01 06:30:00", tz="UTC")

def test_parse_timestamps_no_offset():
    df = pd.DataFrame({
        "start_time": ["2024-01-01 12:00:00.000"],
        "end_time": ["2024-01-01 13:00:00.000"],
    })
    parsed = parse_timestamps(df)
    assert parsed.iloc[0]["start_time"] == pd.Timestamp("2024-01-01 12:00:00", tz="UTC")

def test_parse_timestamps_invalid_date():
    df = pd.DataFrame({
        "start_time": ["invalid"],
    })
    parsed = parse_timestamps(df)
    assert pd.isna(parsed.iloc[0]["start_time"])

def test_resolve_binning_path():
    path = resolve_binning_path("/data", "com.samsung.health.heart_rate", "12345678-uuid.json")
    assert str(path) == "/data/jsons/com.samsung.health.heart_rate/1/12345678-uuid.json"

def test_load_binning_json(tmp_path):
    f = tmp_path / "data.json"
    data = [{"start_time": 1704067200000, "heart_rate": 75}]  # 2024-01-01 00:00:00 UTC
    f.write_text(json.dumps(data))

    df = load_binning_json(f)
    assert len(df) == 1
    assert df.iloc[0]["start_time"] == pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    assert df.iloc[0]["heart_rate"] == 75

def test_load_binning_json_not_found(tmp_path):
    with pytest.warns(UserWarning):
        df = load_binning_json(tmp_path / "missing.json")
    assert df.empty

def test_load_binning_json_bad_json(tmp_path):
    f = tmp_path / "bad.json"
    f.write_text("invalid json")
    with pytest.warns(UserWarning):
        df = load_binning_json(f)
    assert df.empty

def test_coerce_date():
    assert coerce_date(None) is None
    ts = coerce_date("2024-01-01")
    assert ts == pd.Timestamp("2024-01-01", tz="UTC")

    dt = datetime(2024, 1, 1, 12, tzinfo=timezone.utc)
    ts2 = coerce_date(dt)
    assert ts2 == pd.Timestamp("2024-01-01 12:00:00", tz="UTC")

def test_filter_date_range_empty():
    df = pd.DataFrame()
    assert filter_date_range(df).empty

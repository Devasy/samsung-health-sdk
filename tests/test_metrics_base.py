import pytest
import pandas as pd
import json
from pathlib import Path

from samsung_health_sdk.metrics.base import BaseMetric

class DummyMetric(BaseMetric):
    metric_name = "com.samsung.health.dummy"
    value_columns = ["val1", "val2"]

def test_base_metric_not_found(tmp_path):
    m = DummyMetric(tmp_path)
    assert m.available is False
    assert m.load_summary().empty
    assert m.load_detail().empty

def test_base_metric_load_summary(tmp_path):
    f = tmp_path / "com.samsung.health.dummy.123.csv"
    f.write_text("start_time,val1,val2,val3\n2024-01-01 10:00:00,10,20,30")

    m = DummyMetric(tmp_path)
    assert m.available is True

    df = m.load_summary()
    assert len(df) == 1
    # Check that columns were selected
    assert "val1" in df.columns
    assert "val2" in df.columns
    assert "val3" not in df.columns
    assert "start_time" in df.columns

    # Test caching
    m._csv_path = tmp_path / "doesntexist.csv"
    df2 = m.load_summary()
    assert len(df2) == 1

def test_base_metric_filter_dates(tmp_path):
    f = tmp_path / "com.samsung.health.dummy.123.csv"
    f.write_text("start_time,val1\n2024-01-01 10:00:00,10\n2024-01-02 10:00:00,20")

    m = DummyMetric(tmp_path)
    df = m.load_summary(start="2024-01-02")
    assert len(df) == 1
    assert df.iloc[0]["val1"] == 20

def test_base_metric_load_detail(tmp_path):
    f = tmp_path / "com.samsung.health.dummy.123.csv"
    f.write_text("start_time,val1,binning_data,datauuid\n2024-01-01 10:00:00,10,a_binning.json,uuid1")

    # Create binning file
    jsons_dir = tmp_path / "jsons" / "com.samsung.health.dummy" / "a"
    jsons_dir.mkdir(parents=True)
    json_path = jsons_dir / "a_binning.json"
    json_path.write_text(json.dumps([{"start_time": 1704067200000, "val1": 5}]))

    m = DummyMetric(tmp_path)
    df = m.load_detail()
    assert len(df) == 1
    assert df.iloc[0]["val1"] == 5
    assert df.iloc[0]["datauuid"] == "uuid1"

def test_base_metric_load_detail_no_binning(tmp_path):
    f = tmp_path / "com.samsung.health.dummy.123.csv"
    f.write_text("start_time,val1\n2024-01-01 10:00:00,10")

    m = DummyMetric(tmp_path)
    df = m.load_detail()
    assert len(df) == 1
    assert "binning_data" not in df.columns
    assert df.iloc[0]["val1"] == 10

def test_base_metric_load_detail_empty_binning(tmp_path):
    f = tmp_path / "com.samsung.health.dummy.123.csv"
    f.write_text("start_time,val1,binning_data,datauuid\n2024-01-01 10:00:00,10,b_binning.json,uuid1\n2024-01-02 10:00:00,20,c_binning.json,uuid2")

    # only one json exists
    jsons_dir = tmp_path / "jsons" / "com.samsung.health.dummy" / "b"
    jsons_dir.mkdir(parents=True)
    json_path = jsons_dir / "b_binning.json"
    json_path.write_text(json.dumps([{"start_time": 1704067200000, "val1": 5}]))

    m = DummyMetric(tmp_path)
    # the second row fails to parse / is ignored, only 1 output
    with pytest.warns(UserWarning):
        df = m.load_detail()
    assert len(df) == 1

def test_base_metric_find_csv_multiple(tmp_path):
    import time
    f1 = tmp_path / "com.samsung.health.dummy.1.csv"
    f1.write_text("a")
    time.sleep(0.01)
    f2 = tmp_path / "com.samsung.health.dummy.2.csv"
    f2.write_text("b")

    m = DummyMetric(tmp_path)
    assert m._csv_path == f2

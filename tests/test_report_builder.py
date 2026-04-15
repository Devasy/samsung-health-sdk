import pytest
import pandas as pd
import datetime
import numpy as np
from samsung_health_sdk.report.builder import ReportBuilder, _to_records, _Enc
import json
from pathlib import Path

class MockEngine:
    def sleep_sessions(self, start=None, end=None):
        return pd.DataFrame({"date": ["2024-01-01"], "total_h": [8.0]})

    def nightly_physiology(self, start=None, end=None):
        return pd.DataFrame()

    def hrv_readiness(self, start=None, end=None):
        return pd.DataFrame()

    def stress_impact_on_sleep(self, start=None, end=None):
        return pd.DataFrame()

    def daily_activity_profile(self, start=None, end=None):
        return pd.DataFrame()

    def walking_cardiac_load(self, start=None, end=None, source="auto"):
        return pd.DataFrame({"date": ["2024-01-01"], "cardiac_load": [10.0], "rolling_cardiac_load": [10.0], "cardiac_load_trend": [1.0]})

    def daily_hr_stats(self, start=None, end=None):
        return pd.DataFrame()

def test_enc_types():
    assert json.dumps(np.int64(10), cls=_Enc) == "10"
    assert json.dumps(np.float64(10.5), cls=_Enc) == "10.5"
    assert json.dumps(np.bool_(True), cls=_Enc) == "true"
    d = datetime.date(2024, 1, 1)
    assert json.dumps(d, cls=_Enc) == '"2024-01-01"'
    dt = datetime.datetime(2024, 1, 1, 12, 0)
    assert json.dumps(dt, cls=_Enc) == '"2024-01-01T12:00:00"'

def test_to_records():
    # If we pass objects that pd resolves to NaN inside .where(pd.notna(df), None)
    # df.where(pd.notna(df), None) sets None.
    # But wait, python's json serializer can't serialize actual python `nan` to None without mapping, but `None` is mapped to `null`.
    # Let's see what `_to_records` produces.
    # In `_to_records` it does:
    # df = df.where(pd.notna(df), other=None)
    # For float arrays with NaN, using `.where(..., other=None)` turns the column into `object` dtype where missing is `None`.
    # Let's test it using Float64 dtype which pandas properly handles.
    df = pd.DataFrame({
        "a": pd.array([1.0, pd.NA], dtype="Float64"),
    })

    records = _to_records(df)
    assert records[0]["a"] == 1.0
    assert records[1]["a"] is None

def test_builder(tmp_path):
    eng = MockEngine()
    builder = ReportBuilder(eng)
    out = tmp_path / "report.html"

    import samsung_health_sdk.report.builder as rb
    original_template = rb._TEMPLATE
    temp_template = tmp_path / "template.html"
    temp_template.write_text("<html>/*__HEALTH_DATA__*/null/**/</html>")
    rb._TEMPLATE = temp_template

    try:
        builder.build(out)
        assert out.exists()
    finally:
        rb._TEMPLATE = original_template

def test_builder_missing_placeholder(tmp_path):
    eng = MockEngine()
    builder = ReportBuilder(eng)
    out = tmp_path / "report.html"

    import samsung_health_sdk.report.builder as rb
    original_template = rb._TEMPLATE
    temp_template = tmp_path / "template.html"
    temp_template.write_text("<html>bad template</html>")
    rb._TEMPLATE = temp_template

    try:
        with pytest.raises(ValueError, match="missing the __HEALTH_DATA__ placeholder"):
            builder.build(out)
    finally:
        rb._TEMPLATE = original_template

"""Smoke tests for samsung-health-sdk utilities."""

import datetime

import numpy as np
import pandas as pd

from samsung_health_sdk.utils import filter_date_range
from samsung_health_sdk.report.builder import _Enc
import json


# ── filter_date_range ──────────────────────────────────────────────────────


def _make_df(dates):
    return pd.DataFrame(
        {
            "start_time": pd.to_datetime(dates, utc=True),
            "value": range(len(dates)),
        }
    )


def test_filter_date_range_no_bounds():
    df = _make_df(["2024-01-01", "2024-06-15", "2024-12-31"])
    assert len(filter_date_range(df, None, None)) == 3


def test_filter_date_range_start_only():
    df = _make_df(["2024-01-01", "2024-06-15", "2024-12-31"])
    result = filter_date_range(df, "2024-06-01", None)
    assert result["start_time"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-06-15",
        "2024-12-31",
    ]


def test_filter_date_range_end_only():
    df = _make_df(["2024-01-01", "2024-06-15", "2024-12-31"])
    result = filter_date_range(df, None, "2024-06-30")
    assert len(result) == 2


def test_filter_date_range_both_bounds():
    df = _make_df(["2024-01-01", "2024-06-15", "2024-12-31"])
    result = filter_date_range(df, "2024-02-01", "2024-11-30")
    assert result["start_time"].dt.strftime("%Y-%m-%d").tolist() == ["2024-06-15"]
    # inclusive bounds check
    edge = filter_date_range(df, "2024-06-15", "2024-06-15")
    assert edge["start_time"].dt.strftime("%Y-%m-%d").tolist() == ["2024-06-15"]


def test_filter_date_range_empty_df():
    df = pd.DataFrame({"start_time": pd.Series([], dtype="datetime64[ns, UTC]"), "value": []})
    assert filter_date_range(df, "2024-01-01", "2024-12-31").empty


# ── JSON encoder ───────────────────────────────────────────────────────────


def test_enc_numpy_int():
    assert json.dumps(np.int64(42), cls=_Enc) == "42"


def test_enc_numpy_float():
    assert json.dumps(np.float64(3.14), cls=_Enc) == "3.14"


def test_enc_date():
    d = datetime.date(2024, 6, 15)
    assert json.dumps(d, cls=_Enc) == '"2024-06-15"'


def test_enc_numpy_bool():
    assert json.dumps(np.bool_(True), cls=_Enc) == "true"


def test_enc_nan_via_dataframe():
    """NaN values in float columns become None (null) after JSON round-trip."""
    from samsung_health_sdk.report.builder import _to_records

    df = pd.DataFrame({"date": ["2024-01-01"], "val": pd.array([pd.NA], dtype="Float64")})
    records = _to_records(df)
    assert records[0]["val"] is None


def test_enc_dataframe_roundtrip():
    """_to_records produces JSON-safe dicts for normal numeric data."""
    from samsung_health_sdk.report.builder import _to_records

    df = pd.DataFrame({"date": ["2024-01-01", "2024-01-02"], "hr": [72.5, 68.0]})
    records = _to_records(df)
    assert len(records) == 2
    assert records[0]["hr"] == 72.5

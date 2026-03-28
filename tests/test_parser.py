import pytest
import pandas as pd
from pathlib import Path
from samsung_health_sdk import SamsungHealthParser

def test_parser_init_not_found():
    with pytest.raises(FileNotFoundError):
        SamsungHealthParser("non_existent_dir_12345")

def test_parser_list_metrics(tmp_path):
    (tmp_path / "com.samsung.health.metric1.1234567890.csv").touch()
    (tmp_path / "com.samsung.health.metric2.1234567891.csv").touch()
    (tmp_path / "not_a_metric.txt").touch()

    parser = SamsungHealthParser(tmp_path)
    metrics = parser.list_metrics()
    assert metrics == ["com.samsung.health.metric1", "com.samsung.health.metric2"]

    assert parser.has_metric("com.samsung.health.metric1")
    assert not parser.has_metric("com.samsung.health.metric3")

def test_get_metric_not_found(tmp_path):
    parser = SamsungHealthParser(tmp_path)
    with pytest.raises(Exception): # MetricNotFoundError
        parser.get_metric("com.samsung.health.metric1")

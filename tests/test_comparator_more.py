import pytest
import pandas as pd
from samsung_health_sdk.comparator import SamsungHealthComparator

class MockParser:
    def __init__(self, persons_data):
        self.data = persons_data

    def get_heart_rate(self, start=None, end=None, granularity="summary"):
        return self.data.get("heart_rate", pd.DataFrame())

    def get_sleep(self, start=None, end=None):
        return self.data.get("sleep", pd.DataFrame())

    def get_stress(self, start=None, end=None):
        return self.data.get("stress", pd.DataFrame())

    def get_steps(self, start=None, end=None):
        return self.data.get("steps", pd.DataFrame())

    def get_metric(self, metric, start=None, end=None, load_binning=False):
        return self.data.get("metric", pd.DataFrame())

def test_compare_typed_methods_time_shift():
    data = pd.DataFrame({"start_time": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")], "val": [70, 75]})
    comp = SamsungHealthComparator({"p1": MockParser({"heart_rate": data, "sleep": data, "stress": data, "steps": data})})

    assert "relative_time" in comp.compare_heart_rate(time_shift=True).columns
    assert "relative_time" in comp.compare_sleep(time_shift=True).columns
    assert "relative_time" in comp.compare_stress(time_shift=True).columns
    assert "relative_time" in comp.compare_steps(time_shift=True).columns

def test_compare_metric_missing_person():
    comp = SamsungHealthComparator({"p1": MockParser({})})
    with pytest.raises(KeyError):
        comp.compare_metric("some_metric", persons=["p2"])

def test_compare_typed_empty():
    comp = SamsungHealthComparator({"p1": MockParser({})})
    assert comp.compare_heart_rate().empty
    assert comp.compare_sleep().empty
    assert comp.compare_stress().empty
    assert comp.compare_steps().empty

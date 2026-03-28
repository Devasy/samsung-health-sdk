import pytest
import pandas as pd
from samsung_health_sdk.comparator import SamsungHealthComparator

class MockParser:
    def __init__(self, persons_data):
        self.data = persons_data

    def get_metric(self, metric, start=None, end=None, load_binning=False):
        return self.data.get("metric", pd.DataFrame())

    def get_heart_rate(self, start=None, end=None, granularity="summary"):
        return self.data.get("heart_rate", pd.DataFrame())

    def get_sleep(self, start=None, end=None):
        return self.data.get("sleep", pd.DataFrame())

    def get_stress(self, start=None, end=None):
        return self.data.get("stress", pd.DataFrame())

    def get_steps(self, start=None, end=None):
        return self.data.get("steps", pd.DataFrame())

def test_comparator_init():
    with pytest.raises(ValueError):
        SamsungHealthComparator({})

    comp = SamsungHealthComparator({"p1": MockParser({}), "p2": MockParser({})})
    assert comp.persons == ["p1", "p2"]

def test_compare_metric():
    p1_data = pd.DataFrame({"start_time": [pd.Timestamp("2024-01-01 10:00:00")], "val": [10]})
    p2_data = pd.DataFrame({"start_time": [pd.Timestamp("2024-01-02 10:00:00")], "val": [20]})

    comp = SamsungHealthComparator({"p1": MockParser({"metric": p1_data}), "p2": MockParser({"metric": p2_data})})

    df = comp.compare_metric("some_metric")
    assert len(df) == 2
    assert "person" in df.columns
    assert list(df["person"]) == ["p1", "p2"]

def test_compare_metric_time_shift():
    p1_data = pd.DataFrame({"start_time": [pd.Timestamp("2024-01-01 10:00:00"), pd.Timestamp("2024-01-02 10:00:00")], "val": [10, 15]})
    p2_data = pd.DataFrame({"start_time": [pd.Timestamp("2024-01-03 10:00:00"), pd.Timestamp("2024-01-04 10:00:00")], "val": [20, 25]})

    comp = SamsungHealthComparator({"p1": MockParser({"metric": p1_data}), "p2": MockParser({"metric": p2_data})})

    df = comp.compare_metric("some_metric", time_shift=True)
    assert len(df) == 4
    assert "relative_time" in df.columns
    # p1 first row should be timedelta(0)
    assert df[(df["person"] == "p1") & (df["val"] == 10)]["relative_time"].iloc[0] == pd.Timedelta(0)
    # p2 first row should also be timedelta(0)
    assert df[(df["person"] == "p2") & (df["val"] == 20)]["relative_time"].iloc[0] == pd.Timedelta(0)

def test_compare_metric_empty():
    comp = SamsungHealthComparator({"p1": MockParser({}), "p2": MockParser({})})
    df = comp.compare_metric("some_metric")
    assert df.empty

def test_compare_typed_methods():
    hr_data = pd.DataFrame({"start_time": [pd.Timestamp("2024-01-01")], "hr": [70]})
    comp = SamsungHealthComparator({"p1": MockParser({"heart_rate": hr_data, "sleep": hr_data, "stress": hr_data, "steps": hr_data})})

    assert not comp.compare_heart_rate().empty
    assert not comp.compare_sleep().empty
    assert not comp.compare_stress().empty
    assert not comp.compare_steps().empty

import pytest
import pandas as pd
from pathlib import Path
from unittest import mock
from samsung_health_sdk import SamsungHealthParser

@mock.patch("samsung_health_sdk.parser.HeartRateMetric")
@mock.patch("samsung_health_sdk.parser.SleepStageMetric")
@mock.patch("samsung_health_sdk.parser.SkinTemperatureMetric")
@mock.patch("samsung_health_sdk.parser.StressMetric")
@mock.patch("samsung_health_sdk.parser.SpO2Metric")
@mock.patch("samsung_health_sdk.parser.HRVMetric")
@mock.patch("samsung_health_sdk.parser.StepsMetric")
@mock.patch("samsung_health_sdk.parser.RespiratoryRateMetric")
@mock.patch("samsung_health_sdk.parser.ExerciseMetric")
@mock.patch("samsung_health_sdk.parser.MovementMetric")
def test_parser_typed_methods(
    MockMovement, MockExercise, MockRespiratoryRate, MockSteps, MockHRV,
    MockSpO2, MockStress, MockSkinTemp, MockSleep, MockHeartRate, tmp_path
):
    parser = SamsungHealthParser(tmp_path)

    # We create fake csv and mock internal methods
    parser._metric_map["com.samsung.health.heart_rate"] = tmp_path / "com.samsung.health.heart_rate.123.csv"
    parser._metric_map["com.samsung.health.sleep"] = tmp_path / "com.samsung.health.sleep.123.csv"
    parser._metric_map["com.samsung.health.skin_temperature"] = tmp_path / "com.samsung.health.skin_temperature.123.csv"
    parser._metric_map["com.samsung.health.stress"] = tmp_path / "com.samsung.health.stress.123.csv"
    parser._metric_map["com.samsung.health.spo2"] = tmp_path / "com.samsung.health.spo2.123.csv"
    parser._metric_map["com.samsung.health.hrv"] = tmp_path / "com.samsung.health.hrv.123.csv"
    parser._metric_map["com.samsung.health.step_daily_trend"] = tmp_path / "com.samsung.health.step_daily_trend.123.csv"
    parser._metric_map["com.samsung.health.respiratory_rate"] = tmp_path / "com.samsung.health.respiratory_rate.123.csv"
    parser._metric_map["com.samsung.health.exercise"] = tmp_path / "com.samsung.health.exercise.123.csv"
    parser._metric_map["com.samsung.health.movement"] = tmp_path / "com.samsung.health.movement.123.csv"

    # Let's mock the actual _get_or_create_metric to return a dummy with mocked load_summary / load_detail
    class MockMetricInstance:
        def __init__(self, name):
            self.metric_name = name
        def load_summary(self, start=None, end=None):
            return pd.DataFrame({"type": ["summary"], "metric": [self.metric_name]})
        def load_detail(self, start=None, end=None):
            return pd.DataFrame({"type": ["detail"], "metric": [self.metric_name]})

    # Setup the mock class returns
    MockHeartRate.return_value = MockMetricInstance("com.samsung.health.heart_rate")
    MockSleep.return_value = MockMetricInstance("com.samsung.health.sleep")
    MockSkinTemp.return_value = MockMetricInstance("com.samsung.health.skin_temperature")
    MockStress.return_value = MockMetricInstance("com.samsung.health.stress")
    MockSpO2.return_value = MockMetricInstance("com.samsung.health.spo2")
    MockHRV.return_value = MockMetricInstance("com.samsung.health.hrv")
    MockSteps.return_value = MockMetricInstance("com.samsung.health.step_daily_trend")
    MockRespiratoryRate.return_value = MockMetricInstance("com.samsung.health.respiratory_rate")
    MockExercise.return_value = MockMetricInstance("com.samsung.health.exercise")
    MockMovement.return_value = MockMetricInstance("com.samsung.health.movement")

    assert parser.get_heart_rate().iloc[0]["type"] == "summary"
    assert parser.get_heart_rate(granularity="minute").iloc[0]["type"] == "detail"

    assert parser.get_sleep().iloc[0]["type"] == "summary"

    assert parser.get_skin_temperature().iloc[0]["type"] == "summary"
    assert parser.get_skin_temperature(granularity="minute").iloc[0]["type"] == "detail"

    assert parser.get_stress().iloc[0]["type"] == "summary"
    assert parser.get_spo2().iloc[0]["type"] == "summary"

    assert parser.get_hrv(load_binning=False).iloc[0]["type"] == "summary"
    assert parser.get_hrv(load_binning=True).iloc[0]["type"] == "detail"

    assert parser.get_steps().iloc[0]["type"] == "summary"

    assert parser.get_respiratory_rate().iloc[0]["type"] == "summary"
    assert parser.get_respiratory_rate(granularity="minute").iloc[0]["type"] == "detail"

    assert parser.get_exercise().iloc[0]["type"] == "summary"
    assert parser.get_movement().iloc[0]["type"] == "detail"

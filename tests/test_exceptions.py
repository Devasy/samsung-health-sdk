import pytest
from samsung_health_sdk.exceptions import MetricNotFoundError

def test_metric_not_found_error():
    err = MetricNotFoundError("com.missing.metric", ["com.existing.metric"])
    assert "com.missing.metric" in str(err)
    assert "com.existing.metric" in str(err)

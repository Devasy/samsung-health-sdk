"""
samsung-health-sdk
==================
Parse and analyse Samsung Health export data as pandas DataFrames.

Quick start::

    from samsung_health_sdk import SamsungHealthParser, SamsungHealthComparator

    p = SamsungHealthParser("path/to/export_dir")
    print(p.list_metrics())

    hr    = p.get_heart_rate("2024-10-01", "2024-10-31")
    sleep = p.get_sleep("2024-10-01", "2024-10-31")
    steps = p.get_steps("2024-10-01", "2024-10-31")

    # Compare two people
    p2   = SamsungHealthParser("path/to/other_export")
    comp = SamsungHealthComparator({"Alice": p, "Bob": p2})
    df   = comp.compare_heart_rate("2024-10-01", "2024-10-31", time_shift=True)
"""

from samsung_health_sdk.parser import SamsungHealthParser
from samsung_health_sdk.comparator import SamsungHealthComparator
from samsung_health_sdk.features import HealthFeatureEngine
from samsung_health_sdk.exceptions import MetricNotFoundError, DataParseError, SamsungHealthError

__version__ = "0.2.0"
__all__ = [
    "SamsungHealthParser",
    "SamsungHealthComparator",
    "HealthFeatureEngine",
    "MetricNotFoundError",
    "DataParseError",
    "SamsungHealthError",
]

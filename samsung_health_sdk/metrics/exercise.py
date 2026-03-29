"""Exercise session metric parser."""

from __future__ import annotations

import pandas as pd

from samsung_health_sdk.metrics.base import BaseMetric
from samsung_health_sdk.utils import DateLike

# Samsung Health exercise type codes → human-readable names
EXERCISE_TYPE_MAP: dict[int, str] = {
    1001: "Walking",
    1002: "Running",
    1003: "Cycling",
    1004: "Hiking",
    1005: "Rock Climbing",
    1006: "Hiking",
    1008: "Mountain Biking",
    1009: "Roller Skating",
    1010: "Other",
    2001: "Aerobics",
    2002: "Aqua Aerobics",
    2003: "Badminton",
    2004: "Baseball",
    2005: "Basketball",
    2006: "Bowling",
    2007: "Boxing",
    2008: "Circuit Training",
    2009: "Cricket",
    2010: "Dance",
    3001: "Swimming",
    3002: "Biking",
    4001: "Pilates",
    4002: "Stretching",
    4003: "Yoga",
    5001: "Gym & Fitness",
    6001: "Weight Training",
    10004: "Elliptical",
    11001: "High Intensity Interval Training",
}


class ExerciseMetric(BaseMetric):
    """
    Parses com.samsung.shealth.exercise.

    Each row is one exercise session with duration, distance, calorie,
    mean/max heart rate, exercise_type, and associated JSON attachments
    (live_data, route, HR zones, etc.).
    """

    metric_name = "com.samsung.shealth.exercise"
    value_columns = [
        "start_time",
        "end_time",
        "exercise_type",
        "duration",
        "distance",
        "calorie",
        "mean_heart_rate",
        "max_heart_rate",
        "min_heart_rate",
        "mean_speed",
        "max_speed",
        "mean_cadence",
        "max_cadence",
        "vo2_max",
        "altitude_gain",
        "altitude_loss",
        "datauuid",
        "deviceuuid",
        "time_offset",
    ]

    def load_summary(self, start: DateLike = None, end: DateLike = None) -> pd.DataFrame:
        df = super().load_summary(start, end)
        numeric_cols = [
            "duration",
            "distance",
            "calorie",
            "mean_heart_rate",
            "max_heart_rate",
            "min_heart_rate",
            "mean_speed",
            "max_speed",
            "mean_cadence",
            "max_cadence",
            "vo2_max",
            "altitude_gain",
            "altitude_loss",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "exercise_type" in df.columns:
            df["exercise_type"] = pd.to_numeric(df["exercise_type"], errors="coerce")
            df["exercise_name"] = df["exercise_type"].map(EXERCISE_TYPE_MAP).fillna("Unknown")
        # duration is stored in milliseconds — add a seconds column
        if "duration" in df.columns:
            df["duration_sec"] = df["duration"] / 1000.0
        return df

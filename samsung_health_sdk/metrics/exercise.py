"""Exercise session metric parser."""

from __future__ import annotations

import io

import pandas as pd

from samsung_health_sdk.metrics.base import BaseMetric
from samsung_health_sdk.utils import (
    DateLike,
    _detect_skip_rows,
    _strip_namespace,
    load_binning_json,
    parse_timestamps,
    resolve_binning_path,
)

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

# exercise_type values that represent running
_RUNNING_TYPES: frozenset[int] = frozenset({1002})


class ExerciseMetric(BaseMetric):
    """
    Parses com.samsung.shealth.exercise.

    Each row is one exercise session with duration, distance, calorie,
    mean/max heart rate, exercise_type, and associated JSON attachments
    (live_data, route, HR zones, etc.).

    Notes
    -----
    The exercise CSV has one more data column than header columns (the raw CSV
    contains 73 header fields but 74 data fields per row).  ``_load_raw``
    detects this automatically and pads the header with a ``_extra_0`` sentinel
    so that all named columns align correctly.
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

    # ------------------------------------------------------------------
    # CSV loading — override to fix column-count mismatch
    # ------------------------------------------------------------------

    def _load_raw(self) -> pd.DataFrame:
        """
        Read and cache the exercise CSV.

        The exercise CSV header has fewer columns than the data rows (due to an
        extra trailing column added by Samsung Health).  We detect the delta and
        pad the header names so every named column aligns correctly.
        """
        if self._summary_cache is not None:
            return self._summary_cache
        if self._csv_path is None:
            return pd.DataFrame()

        skiprows = _detect_skip_rows(self._csv_path)
        with self._csv_path.open(encoding="utf-8-sig") as fh:
            all_lines = fh.readlines()

        header_line = all_lines[skiprows]
        col_names = [_strip_namespace(c.strip()) for c in header_line.rstrip("\n").split(",")]

        data_lines = all_lines[skiprows + 1 :]
        if data_lines:
            # Sample the first non-empty data line to detect any extra columns
            sample = next((ln for ln in data_lines if ln.strip()), "")
            sample_count = len(sample.rstrip("\n").split(","))
            if sample_count > len(col_names):
                n_extra = sample_count - len(col_names)
                col_names += [f"_extra_{i}" for i in range(n_extra)]

        df = pd.read_csv(
            io.StringIO("".join(data_lines)),
            header=None,
            names=col_names,
            on_bad_lines="skip",
            low_memory=False,
        )
        df = parse_timestamps(df)
        self._summary_cache = df
        return df

    # ------------------------------------------------------------------
    # Summary with derived metrics
    # ------------------------------------------------------------------

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

        # duration_sec / duration_min
        if "duration" in df.columns:
            df["duration_sec"] = df["duration"] / 1000.0
            df["duration_min"] = df["duration_sec"] / 60.0

        # distance_km
        if "distance" in df.columns:
            df["distance_km"] = df["distance"] / 1000.0

        # speed_kmh  (mean_speed is stored in m/s)
        if "mean_speed" in df.columns:
            df["speed_kmh"] = df["mean_speed"] * 3.6

        # pace_min_per_km  (min/km; lower = faster)
        if "duration_min" in df.columns and "distance_km" in df.columns:
            df["pace_min_per_km"] = df["duration_min"] / df["distance_km"].replace(0, float("nan"))

        # beats_per_km — total heartbeats per kilometre covered
        #   = mean_hr (bpm) × duration_min / distance_km
        # A lower value means less cardiac effort per unit distance (like an
        # efficiency or aerobic fitness index).
        if (
            "mean_heart_rate" in df.columns
            and "duration_min" in df.columns
            and "distance_km" in df.columns
        ):
            df["beats_per_km"] = (
                df["mean_heart_rate"]
                * df["duration_min"]
                / df["distance_km"].replace(0, float("nan"))
            )

        return df

    # ------------------------------------------------------------------
    # Running-specific helpers
    # ------------------------------------------------------------------

    def load_runs(
        self,
        start: DateLike = None,
        end: DateLike = None,
        min_distance_km: float = 0.5,
    ) -> pd.DataFrame:
        """
        Return only outdoor running sessions (exercise_type == 1002), deduplicated.

        When the same run is recorded by both Samsung Health and a companion app
        (e.g. Google Fit), this method keeps the record with the richer data
        (prefers non-null mean_heart_rate, then longer reported distance).

        Parameters
        ----------
        start, end:
            Optional date bounds.
        min_distance_km:
            Drop sessions shorter than this (filters out GPS glitches / stray
            sub-minute recordings).

        Returns
        -------
        pd.DataFrame
            One row per unique running session, sorted ascending by start_time.
            Columns include all load_summary() columns plus:
            duration_min, distance_km, speed_kmh, pace_min_per_km, beats_per_km.
        """
        df = self.load_summary(start, end)
        if df.empty:
            return df

        # Filter to running type
        runs = df[df["exercise_type"].isin(_RUNNING_TYPES)].copy()

        # Drop GPS glitch / trivially short entries
        if "distance_km" in runs.columns:
            runs = runs[runs["distance_km"].fillna(0) >= min_distance_km]

        if runs.empty:
            return runs

        # Deduplicate: group sessions that start within 5 minutes of each other.
        # Within each group, prefer the record with HR data over one without.
        runs["_rounded_start"] = runs["start_time"].dt.round("5min")
        runs["_has_hr"] = runs["mean_heart_rate"].notna().astype(int)
        runs = runs.sort_values(
            ["_rounded_start", "_has_hr", "mean_heart_rate"],
            ascending=[True, False, False],
            na_position="last",
        )
        runs = runs.drop_duplicates(subset=["_rounded_start"], keep="first")
        runs = runs.drop(columns=["_rounded_start", "_has_hr"])

        return runs.sort_values("start_time").reset_index(drop=True)

    def load_run_livedata(self, datauuid: str) -> pd.DataFrame:
        """
        Load the per-second live-data JSON for a single running session.

        Returns a DataFrame with columns:
            start_time      — UTC timestamp (from epoch ms)
            elapsed_sec     — seconds since the run started
            elapsed_min     — minutes since the run started
            heart_rate      — bpm (may be NaN between HR readings)
            speed           — m/s
            cadence         — steps/min
            distance        — incremental distance (m) in this interval
            calorie         — incremental calories in this interval
            percent_of_vo2max — % (may be sparsely populated)
            beats_per_m     — heart_rate / (60 × speed)  [beats per metre]
                              Lower = more efficient.  NaN when speed == 0.

        Parameters
        ----------
        datauuid : str
            The ``datauuid`` value from a row returned by ``load_runs()``.
        """
        json_filename = f"{datauuid}.com.samsung.health.exercise.live_data.json"
        json_path = resolve_binning_path(self._data_dir, self.metric_name, json_filename)
        df = load_binning_json(json_path)
        if df.empty:
            return df

        numeric_cols = [
            "heart_rate",
            "speed",
            "cadence",
            "distance",
            "calorie",
            "percent_of_vo2max",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "start_time" in df.columns:
            t0 = df["start_time"].iloc[0]
            df["elapsed_sec"] = (df["start_time"] - t0).dt.total_seconds()
            df["elapsed_min"] = df["elapsed_sec"] / 60.0

        # beats_per_m = HR (bps) / speed (m/s)  = (HR/60) / speed
        if "heart_rate" in df.columns and "speed" in df.columns:
            safe_speed = df["speed"].replace(0, float("nan"))
            df["beats_per_m"] = df["heart_rate"] / (60.0 * safe_speed)

        return df.sort_values("start_time").reset_index(drop=True)

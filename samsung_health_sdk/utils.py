"""Low-level utilities: CSV loading, timestamp parsing, JSON path resolution."""

from __future__ import annotations

import json
import re
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union

import pandas as pd

DateLike = Union[str, datetime, pd.Timestamp, None]

# Regex to parse "UTC+0530", "UTC-0800", "UTC+0000"
_TZ_RE = re.compile(r"UTC([+-])(\d{2})(\d{2})")


def _offset_to_timedelta(tz_str: str) -> timedelta:
    """Convert 'UTC+0530' → timedelta(hours=5, minutes=30)."""
    m = _TZ_RE.match(str(tz_str).strip())
    if not m:
        return timedelta(0)
    sign = 1 if m.group(1) == "+" else -1
    hours, minutes = int(m.group(2)), int(m.group(3))
    return timedelta(hours=sign * hours, minutes=sign * minutes)


def _strip_namespace(col: str) -> str:
    """
    Strip Samsung Health namespace prefix from a column name.

    'com.samsung.health.heart_rate.start_time' → 'start_time'
    'start_time'                                → 'start_time'
    """
    return col.split(".")[-1] if "." in col else col


def _detect_skip_rows(path: Path) -> int:
    """
    Detect whether the file's first line is a metadata row that should be skipped.

    Samsung Health CSVs come in two variants:
      - With metadata row: first line is "com.samsung.something,count,version"
      - Without metadata row: first line is already the column headers

    Returns 1 if a metadata row is present, 0 otherwise.
    """
    try:
        with path.open("r", encoding="utf-8-sig") as fh:
            first_line = fh.readline().strip()
        # Metadata rows start with the metric namespace and have few fields
        first_cell = first_line.split(",")[0]
        if first_cell.startswith("com.samsung."):
            return 1
    except OSError:
        pass
    return 0


def read_csv(path: Union[str, Path]) -> pd.DataFrame:
    """
    Read a Samsung Health CSV file.

    - Auto-detects and skips the metadata row when present.
    - Uses the first non-metadata row as column headers.
    - Strips namespace prefixes from all column names.
    - Handles UTF-8 BOM encoding.

    Returns an empty DataFrame (with stripped columns) if the file has no data rows.
    """
    path = Path(path)
    skiprows = _detect_skip_rows(path)
    try:
        df = pd.read_csv(
            path,
            skiprows=skiprows,
            header=0,
            low_memory=False,
            encoding="utf-8-sig",  # handles BOM
            index_col=False,  # data rows may have a trailing comma → prevents column shift
        )
    except Exception as exc:
        from samsung_health_sdk.exceptions import DataParseError

        raise DataParseError(str(path), str(exc)) from exc

    df.columns = [_strip_namespace(c) for c in df.columns]
    return df


def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert 'start_time' and 'end_time' string columns to tz-aware pd.Timestamp (UTC).

    Samsung Health stores timestamps as "YYYY-MM-DD HH:MM:SS.mmm" strings alongside
    a 'time_offset' column like "UTC+0530".  We parse the naive datetime, apply the
    offset, then normalise to UTC so all records are comparable.

    The original string columns are replaced in-place.
    """
    df = df.copy()

    # Pre-compute per-row UTC offset as timedelta64[ns] (vectorised)
    if "time_offset" in df.columns:
        offset_deltas: pd.Series = pd.to_timedelta(
            df["time_offset"].map(
                lambda x: _offset_to_timedelta(x) if pd.notna(x) else timedelta(0)
            )
        )
    else:
        offset_deltas = None

    for col in ("start_time", "end_time", "create_time", "update_time"):
        if col not in df.columns:
            continue
        # pandas 3.0 may use StringDtype instead of object; accept both
        if not (df[col].dtype == object or pd.api.types.is_string_dtype(df[col])):
            continue
        naive = pd.to_datetime(df[col], format="mixed", errors="coerce")
        if offset_deltas is not None:
            shifted = naive - offset_deltas
        else:
            shifted = naive
        df[col] = shifted.dt.tz_localize("UTC")

    return df


def resolve_binning_path(data_dir: Union[str, Path], metric_name: str, filename: str) -> Path:
    """
    Resolve the full filesystem path to a binning JSON file.

    Samsung Health stores binning files at:
        jsons/{metric_name}/{first_hex_char_of_uuid}/{filename}

    The uuid is the portion of the filename before the first '.'.
    """
    data_dir = Path(data_dir)
    uuid_part = filename.split(".")[0]
    hex_char = uuid_part[0].lower()
    return data_dir / "jsons" / metric_name / hex_char / filename


def load_binning_json(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a Samsung Health binning JSON file into a DataFrame.

    - start_time / end_time are converted from Unix milliseconds to UTC datetime.
    - Returns an empty DataFrame on any parse error (with a warning).
    """
    path = Path(path)
    if not path.exists():
        warnings.warn(f"Binning file not found, skipping: {path}", stacklevel=3)
        return pd.DataFrame()
    try:
        with path.open("r", encoding="utf-8-sig") as fh:
            records = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        warnings.warn(f"Could not parse binning file {path}: {exc}", stacklevel=3)
        return pd.DataFrame()

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    for col in ("start_time", "end_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], unit="ms", utc=True)
    return df


def coerce_date(value: DateLike) -> pd.Timestamp | None:
    """Normalise a user-supplied date value to a UTC-aware pd.Timestamp, or None."""
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def filter_date_range(
    df: pd.DataFrame,
    start: DateLike = None,
    end: DateLike = None,
    col: str = "start_time",
) -> pd.DataFrame:
    """
    Filter a DataFrame to rows where `col` falls within [start, end].

    Both bounds are inclusive. None means unbounded.
    """
    if df.empty or col not in df.columns:
        return df

    start_ts = coerce_date(start)
    end_ts = coerce_date(end)

    mask = pd.Series(True, index=df.index)
    if start_ts is not None:
        mask &= df[col] >= start_ts
    if end_ts is not None:
        # include the full end day
        end_day = end_ts + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        mask &= df[col] <= end_day
    return df.loc[mask].copy()

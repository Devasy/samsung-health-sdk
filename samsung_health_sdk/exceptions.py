"""Custom exceptions for samsung-health-sdk."""

from __future__ import annotations


class SamsungHealthError(Exception):
    """Base exception for all SDK errors."""


class MetricNotFoundError(SamsungHealthError):
    """Raised when a requested metric CSV is not present in the export directory."""

    def __init__(self, metric: str, available: list[str]) -> None:
        self.metric = metric
        self.available = available
        super().__init__(
            f"Metric '{metric}' not found in this export.\n"
            f"Available metrics ({len(available)}): {', '.join(available[:10])}"
            + (" ..." if len(available) > 10 else "")
        )


class DataParseError(SamsungHealthError):
    """Raised when a CSV or JSON file cannot be parsed."""

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Failed to parse '{path}': {reason}")

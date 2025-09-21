"""Time window helpers shared between execution engines."""
from __future__ import annotations

from datetime import datetime, time

import pandas as pd

try:  # Python 3.11+ stdlib
    from zoneinfo import ZoneInfo  # type: ignore
except ImportError:  # pragma: no cover - fallback for environments without zoneinfo
    from pytz import timezone as ZoneInfo  # type: ignore


def _parse_time(value: str | time) -> time:
    if isinstance(value, time):
        return value
    if not isinstance(value, str):
        raise TypeError("Trading window times must be provided as HH:MM strings or datetime.time instances")
    hour, minute = value.split(":", 1)
    return time(int(hour), int(minute))


def _ensure_local_timestamp(ts: datetime | pd.Timestamp, tz: str) -> pd.Timestamp:
    if ts is None:
        raise ValueError("Timestamp required for time window evaluation")
    timestamp = pd.Timestamp(ts)
    try:
        tzinfo = ZoneInfo(tz)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Invalid timezone '{tz}'") from exc

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(tzinfo)
    else:
        timestamp = timestamp.tz_convert(tzinfo)
    return timestamp


def _time_to_minutes(value: time) -> int:
    return value.hour * 60 + value.minute


def is_trading_window(
    ts: datetime | pd.Timestamp,
    start: str | time,
    end: str | time,
    tz: str = "America/New_York",
) -> bool:
    """Return True when *ts* falls within the inclusive [start, end] trading window."""
    start_time = _parse_time(start)
    end_time = _parse_time(end)
    ts_local = _ensure_local_timestamp(ts, tz)
    current_minutes = ts_local.hour * 60 + ts_local.minute
    start_minutes = _time_to_minutes(start_time)
    end_minutes = _time_to_minutes(end_time)

    if start_minutes <= end_minutes:
        return start_minutes <= current_minutes <= end_minutes
    # Overnight windows (e.g., 22:00 -> 02:00)
    return current_minutes >= start_minutes or current_minutes <= end_minutes


def is_force_close(
    ts: datetime | pd.Timestamp,
    force_time: str | time,
    tz: str = "America/New_York",
) -> bool:
    """Return True when the timestamp is greater than or equal to the configured force-close time."""
    force = _parse_time(force_time)
    ts_local = _ensure_local_timestamp(ts, tz)
    return (ts_local.hour, ts_local.minute) >= (force.hour, force.minute)


__all__ = ["is_trading_window", "is_force_close"]

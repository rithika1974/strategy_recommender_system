from __future__ import annotations

from datetime import date, datetime

import pandas as pd

REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "open_interest"]


class DataValidationError(ValueError):
    """Raised when a market-data payload fails validation."""


def _parse_date(value: str | date | datetime | pd.Timestamp | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass

    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    if isinstance(value, str):
        return datetime.fromisoformat(value)

    raise TypeError(f"Unsupported date value type: {type(value)!r}")


def validate_daily_frame(df: pd.DataFrame, from_date: str | None = None, to_date: str | None = None) -> dict:
    """Validate a normalized daily market-data frame and return a structured summary."""
    if df is None or df.empty:
        raise DataValidationError("Dataset is empty")

    errors: list[str] = []
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")

    if errors:
        raise DataValidationError("; ".join(errors))

    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        errors.append("timestamp column must be datetime-like")

    if df["timestamp"].isnull().any():
        errors.append("timestamp contains null values")

    if df["timestamp"].duplicated().any():
        errors.append("duplicate timestamps found")

    if not df["timestamp"].is_monotonic_increasing:
        errors.append("timestamp column is not chronological")

    for column in ["open", "high", "low", "close", "volume", "open_interest"]:
        if df[column].isnull().any():
            errors.append(f"{column} contains null values")

    if (df["volume"] < 0).any():
        errors.append("volume cannot be negative")

    if (df["high"] < df["open"]).any():
        errors.append("high cannot be less than open")
    if (df["high"] < df["close"]).any():
        errors.append("high cannot be less than close")
    if (df["low"] > df["open"]).any():
        errors.append("low cannot be greater than open")
    if (df["low"] > df["close"]).any():
        errors.append("low cannot be greater than close")
    if (df["high"] < df["low"]).any():
        errors.append("high cannot be less than low")

    from_dt = _parse_date(from_date)
    to_dt = _parse_date(to_date)
    if from_dt is not None and to_dt is not None and to_dt < from_dt:
        errors.append("Requested date range is invalid: to_date is before from_date")

    if errors:
        raise DataValidationError("; ".join(errors))

    return {
        "valid": True,
        "record_count": int(len(df)),
        "timestamp_start": df["timestamp"].min().isoformat(),
        "timestamp_end": df["timestamp"].max().isoformat(),
    }

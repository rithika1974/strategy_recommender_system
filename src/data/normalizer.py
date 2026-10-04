from __future__ import annotations

from typing import Any

import pandas as pd

REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "open_interest"]


def _coerce_candle_row(candle: Any) -> dict[str, Any]:
    """Convert a single Upstox candle array to a row dictionary."""
    if not isinstance(candle, (list, tuple)):
        raise ValueError("Each candle must be a list or tuple")
    if len(candle) != 7:
        raise ValueError(f"Each candle must contain exactly 7 values; got {len(candle)}")

    timestamp_raw, open_value, high_value, low_value, close_value, volume_value, open_interest_value = candle

    timestamp = pd.to_datetime(timestamp_raw, errors="raise", utc=True)
    row = {
        "timestamp": timestamp,
        "open": float(open_value),
        "high": float(high_value),
        "low": float(low_value),
        "close": float(close_value),
        "volume": float(volume_value),
        "open_interest": float(open_interest_value),
    }
    return row


def normalize_raw_response(payload: dict[str, Any]) -> pd.DataFrame:
    """Convert a raw Upstox response to a normalized DataFrame."""
    if not isinstance(payload, dict):
        raise ValueError("API response must be a dictionary")

    data = payload.get("data")
    if isinstance(data, dict):
        candles = data.get("candles")
    else:
        candles = payload.get("candles")

    if candles is None:
        raise ValueError("API response does not contain a candle list")
    if not isinstance(candles, list):
        raise ValueError("Candle payload is not a list")
    if not candles:
        raise ValueError("No candle rows were returned")

    rows = [_coerce_candle_row(candle) for candle in candles]
    df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    df = df.sort_values("timestamp").reset_index(drop=True)

    for column in ["open", "high", "low", "close", "volume", "open_interest"]:
        df[column] = pd.to_numeric(df[column], errors="raise")

    return df

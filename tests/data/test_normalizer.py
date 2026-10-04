from __future__ import annotations

import pandas as pd

from src.data.normalizer import normalize_raw_response


def test_normalize_candle_response():
    payload = {
        "data": {
            "candles": [
                [
                    "2020-01-01T00:00:00+05:30",
                    100.0,
                    110.0,
                    95.0,
                    108.0,
                    2500,
                    0,
                ]
            ]
        }
    }

    df = normalize_raw_response(payload)

    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume", "open_interest"]
    assert df["timestamp"].dt.tz is not None
    assert df["close"].iloc[0] == 108.0
    assert df["timestamp"].iloc[0].tzinfo is not None

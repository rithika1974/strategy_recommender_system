from __future__ import annotations
import pandas as pd
from src.features.technical import calculate_breakout_features, calculate_returns, calculate_sma

def frame():
    return pd.DataFrame({"close": range(1, 31), "high": range(2, 32), "low": range(0, 30), "volume": [100] * 30})

def test_returns_and_sma():
    assert calculate_returns(frame()).loc[5, "return_1d"] == 6 / 5 - 1
    assert calculate_sma(frame(), 3).iloc[2] == 2

def test_breakout_uses_previous_window():
    data = frame()
    data.loc[20, "close"] = 100
    data.loc[20, "high"] = 100
    assert calculate_breakout_features(data, 20).loc[20, "breakout_flag"] == 1

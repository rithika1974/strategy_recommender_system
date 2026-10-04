from __future__ import annotations
import pandas as pd
import pytest
from src.features.market import calculate_market_features, calculate_nifty_returns

def test_market_returns_and_alignment():
    nifty = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=3), "close": [100, 110, 121]})
    stock = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=3), "close": [50, 55, 66]})
    assert calculate_nifty_returns(nifty).loc[2, "nifty_return_1d"] == pytest.approx(0.1)
    assert pd.isna(calculate_market_features(stock, nifty).loc[2, "relative_strength_20d"])


def test_market_alignment_does_not_use_future_observation():
    nifty = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-03"]), "close": [100, 110]})
    stock = pd.DataFrame({"date": pd.to_datetime(["2020-01-02", "2020-01-03"]), "close": [50, 55]})
    result = calculate_market_features(stock, nifty)
    assert pd.isna(result.loc[0, "nifty_return_1d"])
    assert result.loc[1, "nifty_return_1d"] == pytest.approx(0.1)

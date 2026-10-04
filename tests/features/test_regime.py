from __future__ import annotations
import pandas as pd
from src.features.regime import calculate_market_trend_regime, calculate_volatility_regime

def test_regime_warmup_and_classification():
    nifty = pd.DataFrame({"close": list(range(1, 51)) + [100]})
    vix = pd.DataFrame({"close": [10] * 20 + [20]})
    assert pd.isna(calculate_market_trend_regime(nifty).iloc[0])
    assert calculate_market_trend_regime(nifty).iloc[-1] == "BULL"
    assert calculate_volatility_regime(vix).iloc[-1] == "HIGH"

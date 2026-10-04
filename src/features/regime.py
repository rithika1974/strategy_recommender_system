from __future__ import annotations

import pandas as pd


def calculate_market_trend_regime(nifty_df: pd.DataFrame) -> pd.Series:
    sma = nifty_df["close"].rolling(50).mean()
    return pd.Series(
        ["BULL" if close > average else "BEAR" if close < average else "SIDEWAYS" if pd.notna(average) else pd.NA
         for close, average in zip(nifty_df["close"], sma)],
        index=nifty_df.index,
        name="market_trend_regime",
    )


def calculate_volatility_regime(vix_df: pd.DataFrame) -> pd.Series:
    median = vix_df["close"].rolling(20).median()
    return pd.Series(
        ["HIGH" if value > threshold else "LOW" if pd.notna(threshold) else pd.NA
         for value, threshold in zip(vix_df["close"], median)],
        index=vix_df.index,
        name="volatility_regime",
    )

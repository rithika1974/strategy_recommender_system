from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    return pd.DataFrame({f"return_{n}d": close / close.shift(n) - 1 for n in (1, 5, 20)}, index=df.index)


def calculate_sma(df: pd.DataFrame, window: int) -> pd.Series:
    return df["close"].rolling(window).mean().rename(f"sma_{window}")


def calculate_ema(df: pd.DataFrame, window: int) -> pd.Series:
    return df["close"].ewm(span=window, adjust=False, min_periods=window).mean().rename(f"ema_{window}")


def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    delta = df["close"].diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = gains.rolling(window).mean()
    average_loss = losses.rolling(window).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))
    return rsi.where(average_loss.ne(0), 100).rename(f"rsi_{window}")


def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    macd = calculate_ema(df, fast) - calculate_ema(df, slow)
    macd_signal = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": macd, "macd_signal": macd_signal}, index=df.index)


def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    previous_close = df["close"].shift()
    true_range = pd.concat(
        [df["high"] - df["low"], (df["high"] - previous_close).abs(), (df["low"] - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window).mean().rename(f"atr_{window}")


def calculate_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
    return df["close"].pct_change().rolling(window).std().rename(f"volatility_{window}")


def calculate_volume_ratio(df: pd.DataFrame, window: int = 20) -> pd.Series:
    return (df["volume"] / df["volume"].rolling(window).mean()).rename(f"volume_ratio_{window}")


def calculate_price_vs_sma(df: pd.DataFrame, window: int) -> pd.Series:
    return (df["close"] / calculate_sma(df, window) - 1).rename(f"price_vs_sma{window}")


def calculate_sma_difference(df: pd.DataFrame, short_window: int = 20, long_window: int = 50) -> pd.Series:
    return (calculate_sma(df, short_window) / calculate_sma(df, long_window) - 1).rename(
        f"sma{short_window}_vs_sma{long_window}"
    )


def calculate_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: int = 2) -> pd.DataFrame:
    middle = calculate_sma(df, window)
    deviation = df["close"].rolling(window).std()
    return pd.DataFrame(
        {"bollinger_upper": middle + num_std * deviation, "bollinger_lower": middle - num_std * deviation},
        index=df.index,
    )


def calculate_bollinger_percent_b(df: pd.DataFrame) -> pd.Series:
    bands = calculate_bollinger_bands(df)
    return ((df["close"] - bands["bollinger_lower"]) / (bands["bollinger_upper"] - bands["bollinger_lower"])).rename(
        "bollinger_percent_b"
    )


def calculate_breakout_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    high = df["high"].rolling(window).max().shift()
    low = df["low"].rolling(window).min().shift()
    previous_high = high
    return pd.DataFrame(
        {
            f"high_{window}": high,
            f"low_{window}": low,
            f"distance_from_high_{window}": df["close"] / high - 1,
            f"distance_from_low_{window}": df["close"] / low - 1,
            "breakout_flag": (df["close"] > previous_high).astype("int64").where(previous_high.notna()),
        },
        index=df.index,
    )

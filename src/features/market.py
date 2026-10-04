from __future__ import annotations

import pandas as pd


def _date_index(df: pd.DataFrame) -> pd.Series:
    values = pd.to_datetime(df["date"] if "date" in df else df["timestamp"], utc=True)
    return values.dt.tz_convert("Asia/Kolkata").dt.normalize()


def calculate_nifty_returns(nifty_df: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=nifty_df.index)
    result["nifty_return_1d"] = nifty_df["close"].pct_change()
    result["nifty_return_20d"] = nifty_df["close"].pct_change(20)
    result["date"] = _date_index(nifty_df)
    return result


def calculate_relative_strength(stock_df: pd.DataFrame, nifty_df: pd.DataFrame) -> pd.Series:
    stock = stock_df["close"].pct_change(20).reset_index(drop=True)
    nifty = nifty_df["close"].pct_change(20).reset_index(drop=True)
    return (stock - nifty).set_axis(stock_df.index).rename("relative_strength_20d")


def calculate_market_features(stock_df: pd.DataFrame, nifty_df: pd.DataFrame | None = None, vix_df: pd.DataFrame | None = None) -> pd.DataFrame:
    dates = _date_index(stock_df)
    result = pd.DataFrame({"date": dates}, index=stock_df.index)
    if nifty_df is not None and not nifty_df.empty:
        nifty = calculate_nifty_returns(nifty_df).drop_duplicates("date").set_index("date")
        result["nifty_return_1d"] = dates.map(nifty["nifty_return_1d"])
        result["nifty_return_20d"] = dates.map(nifty["nifty_return_20d"])
        result["relative_strength_20d"] = stock_df["close"].pct_change(20).to_numpy() - dates.map(
            nifty["nifty_return_20d"]
        ).to_numpy()
    else:
        result[["nifty_return_1d", "nifty_return_20d", "relative_strength_20d"]] = pd.NA
    if vix_df is not None and not vix_df.empty:
        vix = vix_df.assign(date=_date_index(vix_df)).drop_duplicates("date").set_index("date")
        result["india_vix"] = dates.map(vix["close"])
    else:
        result["india_vix"] = pd.NA
    return result

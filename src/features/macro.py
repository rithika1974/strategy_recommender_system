from __future__ import annotations

import pandas as pd


def _returns(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().rename("return_1d")


def calculate_usdinr_return(df: pd.DataFrame) -> pd.Series:
    return _returns(df).rename("usdinr_return_1d")


def calculate_brent_return(df: pd.DataFrame) -> pd.Series:
    return _returns(df).rename("brent_return_1d")


def _asof(stock_dates: pd.Series, macro_df: pd.DataFrame, column: str) -> pd.Series:
    if macro_df is None or macro_df.empty or column not in macro_df:
        return pd.Series(pd.NA, index=stock_dates.index)
    observations = macro_df.copy()
    if "date" not in observations and "timestamp" in observations:
        observations["date"] = observations["timestamp"]
    observations["date"] = pd.to_datetime(observations["date"], utc=True).dt.tz_convert("Asia/Kolkata").dt.normalize()
    observations = observations.sort_values("date")[["date", column]]
    dates = pd.DataFrame({
        "date": pd.to_datetime(stock_dates, utc=True).dt.tz_convert("Asia/Kolkata").dt.normalize()
    }).sort_values("date")
    aligned = pd.merge_asof(dates, observations, on="date", direction="backward")
    return aligned[column].set_axis(dates.index).reindex(stock_dates.index)


def align_repo_rate(stock_dates: pd.Series, repo_df: pd.DataFrame) -> pd.Series:
    return _asof(stock_dates, repo_df, "repo_rate").rename("repo_rate")


def align_cpi(stock_dates: pd.Series, cpi_df: pd.DataFrame) -> pd.Series:
    return _asof(stock_dates, cpi_df, "cpi").rename("cpi")


def calculate_macro_features(
    stock_dates: pd.Series,
    usdinr_df: pd.DataFrame | None = None,
    brent_df: pd.DataFrame | None = None,
    repo_df: pd.DataFrame | None = None,
    cpi_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Calculate active V1 macro features.

    Repo rate and CPI remain schema-compatibility columns only. Inputs are
    accepted for backward compatibility but deliberately ignored in V1.
    """
    _ = repo_df, cpi_df
    result = pd.DataFrame(index=stock_dates.index)
    result["usdinr_return_1d"] = pd.NA if usdinr_df is None else _asof(
        stock_dates, usdinr_df.assign(usdinr_return_1d=calculate_usdinr_return(usdinr_df)), "usdinr_return_1d"
    )
    result["brent_return_1d"] = pd.NA if brent_df is None else _asof(
        stock_dates, brent_df.assign(brent_return_1d=calculate_brent_return(brent_df)), "brent_return_1d"
    )
    result["repo_rate"] = pd.NA
    result["cpi"] = pd.NA
    return result

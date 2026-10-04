from __future__ import annotations
import pandas as pd
import pytest
from src.features.macro import align_cpi, calculate_brent_return, calculate_macro_features, calculate_usdinr_return

def test_macro_alignment_uses_only_past_observations():
    observations = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-03"]), "cpi": [100, 101]})
    result = align_cpi(pd.Series(pd.to_datetime(["2020-01-02", "2020-01-03"])), observations)
    assert list(result) == [100, 101]

def test_usdinr_return():
    assert calculate_usdinr_return(pd.DataFrame({"close": [10, 11]})).iloc[1] == pytest.approx(0.1)


def test_market_macro_returns_do_not_use_future_observations():
    dates = pd.Series(pd.to_datetime(["2020-01-02", "2020-01-03"]))
    usdinr = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-03"]), "close": [10, 12]})
    brent = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-03"]), "close": [50, 55]})
    result = calculate_macro_features(dates, usdinr, brent)
    assert pd.isna(result.loc[0, "usdinr_return_1d"])
    assert result.loc[1, "usdinr_return_1d"] == pytest.approx(0.2)
    assert pd.isna(result.loc[0, "brent_return_1d"])
    assert result.loc[1, "brent_return_1d"] == pytest.approx(0.1)


def test_repo_and_cpi_inputs_remain_unused_null_compatibility_columns():
    dates = pd.Series(pd.to_datetime(["2020-01-02", "2020-01-04"]))
    repo = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-03"]), "repo_rate": [5.15, 4.0]})
    cpi = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-05"]), "cpi": [100, 101]})
    result = calculate_macro_features(dates, repo_df=repo, cpi_df=cpi)
    assert result["repo_rate"].isna().all()
    assert result["cpi"].isna().all()

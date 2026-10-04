from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.backtesting.engine import BacktestConfig, SplitBoundary
from src.schemas.analysis import (
    AnalysisHorizon,
    AnalysisRequest,
    AnalysisValidationError,
)
from src.services.api import QuantitativeAPI, ROUTES
from src.services.quantitative import QuantitativeAnalysisService


class FakeSource:
    source_name = "fake_historical_source"
    frequency = "daily"

    def __init__(self, *, empty: bool = False) -> None:
        self.empty = empty

    def list_stocks(self) -> list[dict]:
        return [{"symbol": "RELIANCE", "company": "Reliance Industries"}]

    def stock_exists(self, symbol: str) -> bool:
        return symbol == "RELIANCE"

    def load_features(self, symbol: str) -> pd.DataFrame:
        if self.empty:
            return pd.DataFrame()
        dates = pd.to_datetime(
            ["2025-01-01", "2025-01-02", "2025-01-03", "2026-01-02"]
        )
        return pd.DataFrame(
            {
                "symbol": [symbol] * 4,
                "date": dates,
                "return_20d": [0.10, 0.10, -0.10, 0.20],
                "price_vs_sma20": [0.1, 0.1, -0.1, 0.1],
                "sma20_vs_sma50": [0.1, 0.1, -0.1, 0.1],
                "rsi_14": [20.0, 20.0, 80.0, 50.0],
                "bollinger_percent_b": [-0.2, -0.2, 1.2, 0.5],
                "breakout_flag": [1, 1, 0, 1],
                "distance_from_low_20": [0.1, 0.1, -0.1, 0.1],
                "volume_ratio_20": [3.0, 3.0, 3.0, 3.0],
                "nifty_return_1d": [0.01, 0.01, -0.01, 0.01],
                "nifty_return_20d": [0.1, 0.1, -0.1, 0.1],
                "relative_strength_20d": [0.0, 0.0, 0.0, 0.1],
                "india_vix": [12.0, 13.0, 14.0, 15.0],
                "usdinr_return_1d": [pd.NA] * 4,
                "brent_return_1d": [pd.NA] * 4,
                "repo_rate": [pd.NA] * 4,
                "cpi": [pd.NA] * 4,
                "market_trend_regime": ["BULL", "BULL", "BEAR", "BULL"],
                "volatility_regime": ["LOW", "LOW", "HIGH", "LOW"],
            }
        )

    def load_prices(self, symbol: str) -> pd.DataFrame:
        if self.empty:
            return pd.DataFrame()
        return pd.DataFrame(
            {
                "symbol": [symbol] * 3,
                "timestamp": pd.date_range(
                    "2025-01-01", periods=3, tz="Asia/Kolkata"
                ),
                "open": [100.0, 101.0, 102.0],
                "high": [102.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0],
                "close": [101.0, 102.0, 103.0],
                "volume": [1_000.0] * 3,
                "open_interest": [0.0] * 3,
            }
        )


def config() -> BacktestConfig:
    return BacktestConfig(
        initial_cash=1_000,
        position_size=0.2,
        transaction_cost_bps=0,
        slippage_bps=0,
        max_positions=5,
        annualization_days=252,
        risk_free_rate=0,
        splits={"unused": SplitBoundary(date(2020, 1, 1), date(2025, 12, 31))},
    )


def artifacts(tmp_path: Path) -> tuple[Path, Path]:
    optimization = tmp_path / "optimization.json"
    robustness = tmp_path / "robustness.json"
    selections = {
        "momentum": {
            "selected_params": {"lookback_days": 20, "threshold": 0.05},
            "selected_train": {},
            "selected_validation": {},
        },
        "trend_following": {
            "selected_params": {
                "fast_window": 20,
                "slow_window": 50,
                "crossover_threshold": 0.0,
                "require_regime_confirmation": False,
            },
            "selected_train": {},
            "selected_validation": {},
        },
        "mean_reversion": {
            "selected_params": {
                "rsi_window": 14,
                "oversold_rsi": 25,
                "overbought_rsi": 75,
                "lower_percent_b": -0.1,
                "upper_percent_b": 1.1,
            },
            "selected_train": {},
            "selected_validation": {},
        },
        "breakout": {
            "selected_params": {"lookback_days": 20, "breakout_multiplier": 2.0},
            "selected_train": {},
            "selected_validation": {},
        },
    }
    optimization.write_text(
        json.dumps({"methodology": {}, "selections": selections}), encoding="utf-8"
    )
    robustness.write_text(
        json.dumps(
            {
                "methodology": {},
                "stock_level_results": {
                    name: [{"symbol": "RELIANCE", "metrics": {}}]
                    for name in selections
                },
                "aggregate_stock_breadth": {},
                "parameter_sensitivity": {},
                "hypothesis_assessment": {},
            }
        ),
        encoding="utf-8",
    )
    return optimization, robustness


def service(tmp_path: Path, *, empty: bool = False) -> QuantitativeAnalysisService:
    optimization, robustness = artifacts(tmp_path)
    return QuantitativeAnalysisService(
        FakeSource(empty=empty),
        backtest_config=config(),
        optimization_path=optimization,
        robustness_path=robustness,
    )


def test_analysis_request_validates_stock_and_horizon_domain():
    request = AnalysisRequest(stock=" reliance ", horizon="intraday")
    assert request.stock == "RELIANCE"
    assert request.horizon is AnalysisHorizon.INTRADAY
    for horizon in AnalysisHorizon:
        assert AnalysisRequest(stock="RELIANCE", horizon=horizon).horizon is horizon
    with pytest.raises(AnalysisValidationError):
        AnalysisRequest(stock="RELIANCE", horizon="day_trade")
    with pytest.raises(AnalysisValidationError):
        AnalysisRequest(stock="bad symbol!", horizon="intraday")


def test_strategy_output_schema_is_consistent_for_all_four_families(tmp_path):
    result = service(tmp_path).get_strategies(
        AnalysisRequest(stock="RELIANCE", horizon="intraday")
    )
    assert result["status"] == "ok"
    assert result["source"]["frequency"] == "daily"
    assert result["source"]["horizon_used_to_change_strategy_rules"] is False
    strategies = result["data"]["strategies"]
    assert {item["strategy"] for item in strategies} == {
        "momentum",
        "trend_following",
        "mean_reversion",
        "breakout",
    }
    expected = {
        "strategy",
        "signal",
        "parameters",
        "entry",
        "exit",
        "stop_loss",
        "take_profit",
        "performance",
        "risk_metrics",
        "as_of",
        "status",
        "unavailable_fields",
    }
    assert all(set(item) == expected for item in strategies)
    assert all(item["entry"] is None for item in strategies)
    assert all(item["stop_loss"] is None for item in strategies)
    assert all(
        item["performance"]["period"]["end_date"] == "2025-12-31"
        for item in strategies
    )


def test_valid_and_invalid_stock_are_handled_cleanly(tmp_path):
    api = QuantitativeAPI(service(tmp_path))
    status, body = api.handle(
        "GET", "/features/reliance", {"horizon": "intraday"}
    )
    assert status == 200
    assert body["request"] == {"stock": "RELIANCE", "horizon": "intraday"}
    status, body = api.handle(
        "GET", "/features/UNKNOWN", {"horizon": "intraday"}
    )
    assert status == 404
    assert body["error"] == "unknown_stock"


def test_invalid_or_missing_horizon_returns_bad_request(tmp_path):
    api = QuantitativeAPI(service(tmp_path))
    for query in ({"horizon": "hourly"}, {}):
        status, body = api.handle("GET", "/strategies/RELIANCE", query)
        assert status == 400
        assert body["error"] == "invalid_analysis_request"


def test_missing_data_returns_explicit_empty_responses(tmp_path):
    backend = service(tmp_path, empty=True)
    request = AnalysisRequest(stock="RELIANCE", horizon="swing")
    assert backend.get_features(request)["status"] == "empty"
    assert backend.get_market_context(request)["status"] == "empty"
    assert backend.get_strategies(request)["status"] == "empty"
    assert backend.get_backtest(request)["status"] == "empty"


def test_all_declared_service_routes_return_json_compatible_structure(tmp_path):
    api = QuantitativeAPI(service(tmp_path))
    assert len(ROUTES) == 7
    status, stocks = api.handle("GET", "/stocks")
    assert status == 200
    assert json.loads(json.dumps(stocks))["data"]["count"] == 1
    for endpoint in (
        "features",
        "market-context",
        "strategies",
        "backtest",
        "optimization",
        "robustness",
    ):
        status, body = api.handle(
            "GET", f"/{endpoint}/RELIANCE", {"horizon": "intraday"}
        )
        assert status == 200
        assert set(body) == {"status", "request", "source", "data", "warnings"}
        json.dumps(body, allow_nan=False)


def test_features_exclude_cpi_and_repo_but_market_context_marks_them_unused(tmp_path):
    backend = service(tmp_path)
    request = AnalysisRequest(stock="RELIANCE", horizon="short_term")
    features = backend.get_features(request)["data"]
    assert features["excluded_v1_columns"] == ["cpi", "repo_rate"]
    assert "cpi" not in features["latest"]
    context = backend.get_market_context(request)["data"]
    assert context["unused_v1_features"] == {"repo_rate": None, "cpi": None}


def test_repeated_service_execution_is_deterministic_and_2026_is_not_backtested(tmp_path):
    backend = service(tmp_path)
    request = AnalysisRequest(stock="RELIANCE", horizon="medium_term")
    first = backend.get_backtest(request)
    second = backend.get_backtest(request)
    assert first == second
    assert first["data"]["test_2026_accessed"] is False
    assert all(
        result["period"]["end_date"] == "2025-12-31"
        for result in first["data"]["strategies"].values()
    )

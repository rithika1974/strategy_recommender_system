"""Framework-neutral facade over the existing deterministic quant backend."""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtesting.engine import BacktestConfig, load_backtest_config, run_backtest
from src.evaluation.walk_forward import evaluation_config
from src.schemas.analysis import AnalysisRequest, ServiceResponse
from src.services.source import AnalysisDataSource, SQLiteAnalysisDataSource
from src.strategies.base import BaseStrategy
from src.strategies.registry import STRATEGY_TYPES, load_enabled_strategies


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OPTIMIZATION_PATH = (
    PROJECT_ROOT / "data" / "strategy" / "performance" / "optimization_results.json"
)
DEFAULT_ROBUSTNESS_PATH = (
    PROJECT_ROOT / "data" / "strategy" / "performance" / "robustness_results.json"
)
V1_UNUSED_FEATURES = {"repo_rate", "cpi"}
MARKET_CONTEXT_COLUMNS = (
    "nifty_return_1d",
    "nifty_return_20d",
    "relative_strength_20d",
    "india_vix",
    "usdinr_return_1d",
    "brent_return_1d",
    "market_trend_regime",
    "volatility_regime",
)


class UnknownStockError(ValueError):
    """Raised when a request refers to a stock outside the configured database."""


def _json_safe(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


class QuantitativeAnalysisService:
    """Stable system-level operations intended for future agent consumers."""

    def __init__(
        self,
        data_source: AnalysisDataSource | None = None,
        *,
        backtest_config: BacktestConfig | None = None,
        optimization_path: str | Path = DEFAULT_OPTIMIZATION_PATH,
        robustness_path: str | Path = DEFAULT_ROBUSTNESS_PATH,
    ) -> None:
        self.data_source = data_source or SQLiteAnalysisDataSource()
        self.backtest_config = backtest_config or load_backtest_config()
        self.optimization_path = Path(optimization_path)
        self.robustness_path = Path(robustness_path)

    def _source(self, request: AnalysisRequest | None = None) -> dict[str, Any]:
        result = {
            "name": self.data_source.source_name,
            "frequency": self.data_source.frequency,
            "primary_historical_provider": "upstox",
        }
        if request is not None:
            result.update(
                {
                    "requested_horizon": request.horizon.value,
                    "horizon_used_to_change_strategy_rules": False,
                    "horizon_calibration_status": "not_calibrated_in_v1",
                }
            )
        return result

    def _response(
        self,
        status: str,
        data: Any,
        *,
        request: AnalysisRequest | None = None,
        warnings: list[str] | None = None,
    ) -> dict[str, Any]:
        model = ServiceResponse(
            status=status,
            request=(request.model_dump(mode="json") if request else None),
            source=self._source(request),
            data=_json_safe(data),
            warnings=warnings or [],
        )
        return model.model_dump(mode="json")

    def _validate_stock(self, request: AnalysisRequest) -> None:
        if not self.data_source.stock_exists(request.stock):
            raise UnknownStockError(f"unknown stock: {request.stock}")

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError(f"artifact must contain a JSON object: {path}")
        return value

    def _selected_strategies(self) -> tuple[dict[str, BaseStrategy], str]:
        optimization = self._read_json(self.optimization_path)
        if optimization and isinstance(optimization.get("selections"), dict):
            strategies = {
                name: STRATEGY_TYPES[name](dict(selection["selected_params"]))
                for name, selection in sorted(optimization["selections"].items())
                if name in STRATEGY_TYPES
            }
            if strategies:
                return strategies, "optimization_results"
        return (
            {strategy.name: strategy for strategy in load_enabled_strategies()},
            "baseline_strategies_config",
        )

    def list_stocks(self) -> dict[str, Any]:
        stocks = self.data_source.list_stocks()
        return self._response(
            "ok" if stocks else "empty",
            {"count": len(stocks), "stocks": stocks},
        )

    def get_features(self, request: AnalysisRequest) -> dict[str, Any]:
        self._validate_stock(request)
        features = self.data_source.load_features(request.stock)
        if features.empty:
            return self._response(
                "empty",
                {"symbol": request.stock, "row_count": 0, "latest": None},
                request=request,
                warnings=["No feature rows are available for this stock."],
            )
        latest = features.sort_values("date").iloc[-1]
        active_columns = [
            column
            for column in features.columns
            if column not in {"symbol", "date", *V1_UNUSED_FEATURES}
        ]
        return self._response(
            "ok",
            {
                "symbol": request.stock,
                "row_count": len(features),
                "start_date": str(features["date"].min()),
                "end_date": str(features["date"].max()),
                "active_feature_columns": active_columns,
                "excluded_v1_columns": sorted(V1_UNUSED_FEATURES),
                "latest": {column: latest[column] for column in ["date", *active_columns]},
            },
            request=request,
        )

    def get_market_context(self, request: AnalysisRequest) -> dict[str, Any]:
        self._validate_stock(request)
        features = self.data_source.load_features(request.stock)
        if features.empty:
            return self._response(
                "empty",
                {"symbol": request.stock, "market_context": None},
                request=request,
                warnings=["No feature rows are available for market context."],
            )
        latest = features.sort_values("date").iloc[-1]
        context = {
            column: latest[column] if column in latest.index else None
            for column in MARKET_CONTEXT_COLUMNS
        }
        return self._response(
            "ok",
            {
                "symbol": request.stock,
                "as_of": latest["date"],
                "market_context": context,
                "unused_v1_features": {"repo_rate": None, "cpi": None},
            },
            request=request,
        )

    @staticmethod
    def _pretest(frame: pd.DataFrame, column: str) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()
        dates = pd.to_datetime(frame[column], utc=True, errors="raise")
        if column == "timestamp":
            dates = dates.dt.tz_convert("Asia/Kolkata")
        return frame.loc[dates.dt.date < date(2026, 1, 1)].copy()

    def _stock_backtests(
        self, request: AnalysisRequest
    ) -> tuple[dict[str, Any], dict[str, BaseStrategy], str]:
        features = self._pretest(self.data_source.load_features(request.stock), "date")
        prices = self._pretest(self.data_source.load_prices(request.stock), "timestamp")
        strategies, parameter_source = self._selected_strategies()
        if features.empty or prices.empty:
            return {}, strategies, parameter_source
        config = evaluation_config(
            self.backtest_config,
            date(2020, 1, 1),
            date(2025, 12, 31),
            split_name="analysis",
            max_positions=1,
        )
        output: dict[str, Any] = {}
        for name, strategy in sorted(strategies.items()):
            result = run_backtest(
                strategy.generate_signals(features),
                prices,
                strategy=name,
                split="analysis",
                config=config,
            )
            output[name] = {
                "parameters": dict(strategy.params),
                "metrics": dict(result.metrics),
                "diagnostics": dict(result.diagnostics),
                "period": {"start_date": "2020-01-01", "end_date": "2025-12-31"},
                "ending_position": (
                    "flat"
                    if result.equity_curve.empty
                    else str(result.equity_curve.iloc[-1]["position"])
                ),
            }
        return output, strategies, parameter_source

    def get_backtest(self, request: AnalysisRequest) -> dict[str, Any]:
        self._validate_stock(request)
        results, _, parameter_source = self._stock_backtests(request)
        if not results:
            return self._response(
                "empty",
                {"symbol": request.stock, "strategies": {}},
                request=request,
                warnings=["Features or prices are missing; no backtest was run."],
            )
        return self._response(
            "ok",
            {
                "symbol": request.stock,
                "scope": "single_stock_pretest_diagnostic",
                "parameter_source": parameter_source,
                "execution": "next_open",
                "test_2026_accessed": False,
                "strategies": results,
            },
            request=request,
        )

    def get_strategies(self, request: AnalysisRequest) -> dict[str, Any]:
        self._validate_stock(request)
        features = self.data_source.load_features(request.stock)
        if features.empty:
            return self._response(
                "empty",
                {"symbol": request.stock, "strategies": []},
                request=request,
                warnings=["No features are available for strategy signals."],
            )
        backtests, strategies, parameter_source = self._stock_backtests(request)
        outputs = []
        for name, strategy in sorted(strategies.items()):
            signals = strategy.generate_signals(features)
            latest = signals.sort_values("date").iloc[-1]
            metrics = backtests.get(name, {}).get("metrics", {})
            performance = (
                {
                    "scope": "single_stock_pretest_diagnostic",
                    "period": backtests[name]["period"],
                    "metrics": metrics,
                }
                if name in backtests
                else {
                    "scope": "unavailable",
                    "period": None,
                    "metrics": {},
                }
            )
            risk = {
                key: metrics[key]
                for key in ("volatility", "max_drawdown")
                if key in metrics
            }
            outputs.append(
                strategy.structured_output(
                    str(latest["signal"]),
                    as_of=str(latest["date"]),
                    performance=performance,
                    risk_metrics=risk,
                ).model_dump(mode="json")
            )
        return self._response(
            "ok",
            {
                "symbol": request.stock,
                "parameter_source": parameter_source,
                "strategy_horizon_calibrated": False,
                "strategies": outputs,
            },
            request=request,
            warnings=(
                [
                    "Current V1 signals use daily features and have not been "
                    "calibrated specifically for the requested horizon."
                ]
            ),
        )

    def get_optimization(self, request: AnalysisRequest) -> dict[str, Any]:
        self._validate_stock(request)
        report = self._read_json(self.optimization_path)
        if report is None:
            return self._response(
                "unavailable",
                {"symbol": request.stock, "optimization": None},
                request=request,
                warnings=["Optimization artifact is not available."],
            )
        selections = {
            name: {
                "selected_params": value.get("selected_params"),
                "selected_train": value.get("selected_train"),
                "selected_validation": value.get("selected_validation"),
                "overfitting_flags": value.get("overfitting_flags", []),
            }
            for name, value in sorted(report.get("selections", {}).items())
        }
        return self._response(
            "ok",
            {
                "symbol": request.stock,
                "scope": "universe_level_not_stock_specific",
                "methodology": report.get("methodology"),
                "selections": selections,
            },
            request=request,
        )

    def get_robustness(self, request: AnalysisRequest) -> dict[str, Any]:
        self._validate_stock(request)
        report = self._read_json(self.robustness_path)
        if report is None:
            return self._response(
                "unavailable",
                {"symbol": request.stock, "robustness": None},
                request=request,
                warnings=["Robustness artifact is not available."],
            )
        stock_rows: dict[str, Any] = {}
        for name, rows in sorted(report.get("stock_level_results", {}).items()):
            stock_rows[name] = next(
                (row for row in rows if row.get("symbol") == request.stock), None
            )
        return self._response(
            "ok",
            {
                "symbol": request.stock,
                "scope": "universe_diagnostic_with_stock_breadth_extract",
                "methodology": report.get("methodology"),
                "stock_results": stock_rows,
                "aggregate_stock_breadth": report.get("aggregate_stock_breadth"),
                "parameter_sensitivity": report.get("parameter_sensitivity"),
                "hypothesis_assessment": report.get("hypothesis_assessment"),
            },
            request=request,
        )

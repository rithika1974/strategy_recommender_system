"""Deterministic temporal, parameter, stock, and regime robustness diagnostics."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.backtesting.engine import BacktestConfig, BacktestResult, run_backtest
from src.evaluation.metrics import calculate_max_drawdown, calculate_sharpe
from src.evaluation.optimization import OptimizationConfig
from src.evaluation.sensitivity import run_sensitivity_analysis
from src.evaluation.validation import calculate_objective
from src.evaluation.walk_forward import (
    TemporalFold,
    evaluation_config,
    run_temporal_stability,
    validate_temporal_folds,
)
from src.strategies.base import BaseStrategy
from src.strategies.registry import STRATEGY_TYPES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "robustness.yaml"
CLASSIFICATIONS = {"STABLE", "SENSITIVE", "BOUNDARY", "INSUFFICIENT_EVIDENCE"}


@dataclass(frozen=True)
class SensitivityThresholds:
    minimum_neighbors: int
    minimum_validation_trades: int
    maximum_objective_difference: float
    maximum_return_difference: float
    material_return_reversal: float
    boundary_minimum_grid_values: int


@dataclass(frozen=True)
class RegimeThresholds:
    minimum_observations: int
    minimum_trades: int
    market_values: tuple[str, ...]
    volatility_values: tuple[str, ...]


@dataclass(frozen=True)
class HypothesisThresholds:
    regime_sharpe_spread: float
    stock_concentration_top_fraction: float
    stock_concentration_threshold: float


@dataclass(frozen=True)
class RobustnessConfig:
    test_cutoff: date
    expected_stock_count: int
    output_path: Path
    temporal_folds: tuple[TemporalFold, ...]
    analysis_start_date: date
    analysis_end_date: date
    sensitivity: SensitivityThresholds
    regime: RegimeThresholds
    hypothesis: HypothesisThresholds


def _date(value: Any, label: str) -> date:
    try:
        return pd.Timestamp(value).date()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a valid date") from exc


def load_robustness_config(path: str | Path = DEFAULT_CONFIG_PATH) -> RobustnessConfig:
    """Load and validate all thresholds before analysis is calculated."""
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = (yaml.safe_load(handle) or {}).get("robustness", {})
    folds = tuple(
        TemporalFold(
            name=str(item["name"]),
            context_start_date=_date(item["context_start_date"], "context_start_date"),
            context_end_date=_date(item["context_end_date"], "context_end_date"),
            evaluation_start_date=_date(
                item["evaluation_start_date"], "evaluation_start_date"
            ),
            evaluation_end_date=_date(item["evaluation_end_date"], "evaluation_end_date"),
        )
        for item in raw["temporal_folds"]
    )
    sensitivity_raw = raw["sensitivity"]
    regime_raw = raw["regime"]
    hypothesis_raw = raw["hypothesis"]
    period = raw["analysis_period"]
    config = RobustnessConfig(
        test_cutoff=_date(raw["test_cutoff"], "test_cutoff"),
        expected_stock_count=int(raw["expected_stock_count"]),
        output_path=PROJECT_ROOT / str(raw["output_path"]),
        temporal_folds=folds,
        analysis_start_date=_date(period["start_date"], "analysis.start_date"),
        analysis_end_date=_date(period["end_date"], "analysis.end_date"),
        sensitivity=SensitivityThresholds(
            minimum_neighbors=int(sensitivity_raw["minimum_neighbors"]),
            minimum_validation_trades=int(
                sensitivity_raw["minimum_validation_trades"]
            ),
            maximum_objective_difference=float(
                sensitivity_raw["maximum_objective_difference"]
            ),
            maximum_return_difference=float(
                sensitivity_raw["maximum_return_difference"]
            ),
            material_return_reversal=float(
                sensitivity_raw["material_return_reversal"]
            ),
            boundary_minimum_grid_values=int(
                sensitivity_raw["boundary_minimum_grid_values"]
            ),
        ),
        regime=RegimeThresholds(
            minimum_observations=int(regime_raw["minimum_observations"]),
            minimum_trades=int(regime_raw["minimum_trades"]),
            market_values=tuple(map(str, regime_raw["market_values"])),
            volatility_values=tuple(map(str, regime_raw["volatility_values"])),
        ),
        hypothesis=HypothesisThresholds(
            regime_sharpe_spread=float(hypothesis_raw["regime_sharpe_spread"]),
            stock_concentration_top_fraction=float(
                hypothesis_raw["stock_concentration_top_fraction"]
            ),
            stock_concentration_threshold=float(
                hypothesis_raw["stock_concentration_threshold"]
            ),
        ),
    )
    validate_temporal_folds(config.temporal_folds, config.test_cutoff)
    if config.analysis_start_date > config.analysis_end_date:
        raise ValueError("analysis period ends before it starts")
    if config.analysis_end_date >= config.test_cutoff:
        raise ValueError("analysis period accesses the frozen TEST period")
    if config.expected_stock_count <= 0:
        raise ValueError("expected_stock_count must be positive")
    numeric_thresholds = (
        config.sensitivity.minimum_neighbors,
        config.sensitivity.minimum_validation_trades,
        config.sensitivity.maximum_objective_difference,
        config.sensitivity.maximum_return_difference,
        config.sensitivity.material_return_reversal,
        config.regime.minimum_observations,
        config.regime.minimum_trades,
        config.hypothesis.regime_sharpe_spread,
        config.hypothesis.stock_concentration_top_fraction,
        config.hypothesis.stock_concentration_threshold,
    )
    if any(value < 0 for value in numeric_thresholds):
        raise ValueError("robustness thresholds cannot be negative")
    if not 0 < config.hypothesis.stock_concentration_top_fraction <= 1:
        raise ValueError("stock concentration fraction must be in (0, 1]")
    if not 0 <= config.hypothesis.stock_concentration_threshold <= 1:
        raise ValueError("stock concentration threshold must be in [0, 1]")
    return config


def load_optimization_report(path: str | Path) -> dict[str, Any]:
    """Read selected parameters without altering the optimization artifact."""
    with Path(path).open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(report.get("selections"), dict):
        raise ValueError("optimization report does not contain selections")
    return report


def load_selected_strategies(report: dict[str, Any]) -> dict[str, BaseStrategy]:
    """Instantiate exactly the configurations selected by optimization."""
    selections = report.get("selections", {})
    unknown = set(selections).difference(STRATEGY_TYPES)
    if unknown:
        raise ValueError(f"optimization report contains unknown strategies: {sorted(unknown)}")
    strategies: dict[str, BaseStrategy] = {}
    for name in sorted(selections):
        params = selections[name].get("selected_params")
        if not isinstance(params, dict):
            raise ValueError(f"selected parameters are missing for {name}")
        strategies[name] = STRATEGY_TYPES[name](dict(params))
    if not strategies:
        raise ValueError("optimization report selected no strategies")
    return strategies


def find_neighboring_configurations(
    selected: dict[str, Any],
    baseline: dict[str, Any],
    grid: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    """Return grid candidates one adjacent step away in exactly one parameter."""
    candidates = run_sensitivity_analysis(baseline, grid)
    neighbors: list[dict[str, Any]] = []
    for key in sorted(grid):
        values = list(dict.fromkeys(grid[key]))
        if selected.get(key) not in values:
            raise ValueError(f"selected {key} is outside the existing grid")
        selected_index = values.index(selected[key])
        adjacent = {
            index
            for index in (selected_index - 1, selected_index + 1)
            if 0 <= index < len(values)
        }
        for candidate in candidates:
            changed = [name for name in grid if candidate.get(name) != selected.get(name)]
            if changed == [key] and values.index(candidate[key]) in adjacent:
                neighbors.append(candidate)
    return sorted(
        neighbors,
        key=lambda params: json.dumps(params, sort_keys=True, separators=(",", ":")),
    )


def _boundary_parameters(
    selected: dict[str, Any],
    grid: dict[str, list[Any]],
    minimum_grid_values: int,
) -> list[str]:
    boundary: list[str] = []
    for key, raw_values in sorted(grid.items()):
        values = [
            value
            for value in dict.fromkeys(raw_values)
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        if len(values) >= minimum_grid_values and selected.get(key) in {
            min(values),
            max(values),
        }:
            boundary.append(key)
    return boundary


def classify_sensitivity(
    selected_result: dict[str, Any],
    neighbor_results: list[dict[str, Any]],
    selected_params: dict[str, Any],
    grid: dict[str, list[Any]],
    thresholds: SensitivityThresholds,
) -> tuple[str, list[str]]:
    """Classify with configured thresholds and deterministic precedence."""
    reasons: list[str] = []
    if len(neighbor_results) < thresholds.minimum_neighbors:
        return "INSUFFICIENT_EVIDENCE", ["too_few_comparable_neighbors"]
    all_results = [selected_result, *neighbor_results]
    if any(
        int(result.get("number_of_trades", 0)) < thresholds.minimum_validation_trades
        for result in all_results
    ):
        return "INSUFFICIENT_EVIDENCE", ["too_few_validation_trades"]
    boundary = _boundary_parameters(
        selected_params, grid, thresholds.boundary_minimum_grid_values
    )
    if boundary:
        return "BOUNDARY", [f"selected_{name}_at_grid_boundary" for name in boundary]

    selected_objective = float(selected_result["objective"])
    selected_return = float(selected_result["cumulative_return"])
    for result in neighbor_results:
        objective_difference = abs(float(result["objective"]) - selected_objective)
        return_value = float(result["cumulative_return"])
        return_difference = abs(return_value - selected_return)
        reversal = (
            selected_return * return_value < 0
            and return_difference >= thresholds.material_return_reversal
        )
        if objective_difference > thresholds.maximum_objective_difference:
            reasons.append("material_objective_difference")
        if return_difference > thresholds.maximum_return_difference:
            reasons.append("material_return_difference")
        if reversal:
            reasons.append("material_return_reversal")
    if reasons:
        return "SENSITIVE", sorted(set(reasons))
    return "STABLE", ["neighbors_within_predeclared_thresholds"]


def _same_params(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def run_parameter_sensitivity(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    strategies: dict[str, BaseStrategy],
    optimization_report: dict[str, Any],
    *,
    backtest_config: BacktestConfig,
    optimization_config: OptimizationConfig,
    thresholds: SensitivityThresholds,
) -> dict[str, Any]:
    """Compare selected configurations with existing-grid adjacent candidates."""
    stored = optimization_report.get("candidate_results", [])
    output: dict[str, Any] = {}
    for name in sorted(strategies):
        selection = optimization_report["selections"][name]
        selected_params = dict(selection["selected_params"])
        baseline = dict(selection["baseline_params"])
        grid = optimization_config.parameter_grids[name]
        neighbor_params = find_neighboring_configurations(selected_params, baseline, grid)
        selected_result = dict(selection["selected_validation"])
        neighbor_results: list[dict[str, Any]] = []
        for params in neighbor_params:
            existing = next(
                (
                    record
                    for record in stored
                    if record.get("strategy") == name
                    and record.get("split") == "validation"
                    and _same_params(record.get("params", {}), params)
                ),
                None,
            )
            if existing is not None:
                record = dict(existing)
                source = "stored_validation_result"
            else:
                strategy = STRATEGY_TYPES[name](params)
                result = run_backtest(
                    strategy.generate_signals(features),
                    prices,
                    strategy=name,
                    split="validation",
                    config=backtest_config,
                )
                record = {
                    "strategy": name,
                    "params": params,
                    "split": "validation",
                    "objective": calculate_objective(
                        result.metrics,
                        sharpe_weight=optimization_config.sharpe_weight,
                        return_weight=optimization_config.return_weight,
                        drawdown_weight=optimization_config.drawdown_weight,
                    ),
                    **result.metrics,
                }
                source = "diagnostic_validation_run"
            record["source"] = source
            neighbor_results.append(record)
        classification, reasons = classify_sensitivity(
            selected_result,
            neighbor_results,
            selected_params,
            grid,
            thresholds,
        )
        output[name] = {
            "selected_params": selected_params,
            "selected_validation": selected_result,
            "neighbors": neighbor_results,
            "classification": classification,
            "reasons": reasons,
        }
    return output


def aggregate_stock_results(
    stock_results: list[dict[str, Any]], *, top_fraction: float
) -> dict[str, Any]:
    """Summarize one-symbol backtests without portfolio slot competition."""
    if not stock_results:
        return {
            "number_of_stocks": 0,
            "mean_stock_return": None,
            "median_stock_return": None,
            "mean_sharpe": None,
            "median_sharpe": None,
            "positive_return_percentage": None,
            "positive_sharpe_percentage": None,
            "best_stock": None,
            "worst_stock": None,
            "positive_return_concentration_top_fraction": None,
        }
    ordered = sorted(stock_results, key=lambda row: row["symbol"])
    returns = pd.Series([row["metrics"]["cumulative_return"] for row in ordered], dtype=float)
    sharpes = pd.Series([row["metrics"]["sharpe"] for row in ordered], dtype=float)
    ranked = sorted(
        ordered,
        key=lambda row: (float(row["metrics"]["cumulative_return"]), row["symbol"]),
    )
    positive = sorted((value for value in returns if value > 0), reverse=True)
    concentration: float | None = None
    if positive:
        count = max(1, math.ceil(len(ordered) * top_fraction))
        concentration = float(sum(positive[:count]) / sum(positive))
    return {
        "number_of_stocks": len(ordered),
        "mean_stock_return": float(returns.mean()),
        "median_stock_return": float(returns.median()),
        "mean_sharpe": float(sharpes.mean()),
        "median_sharpe": float(sharpes.median()),
        "positive_return_percentage": float(returns.gt(0).mean() * 100),
        "positive_sharpe_percentage": float(sharpes.gt(0).mean() * 100),
        "best_stock": {
            "symbol": ranked[-1]["symbol"],
            "return": float(ranked[-1]["metrics"]["cumulative_return"]),
        },
        "worst_stock": {
            "symbol": ranked[0]["symbol"],
            "return": float(ranked[0]["metrics"]["cumulative_return"]),
        },
        "positive_return_concentration_top_fraction": concentration,
    }


def run_cross_stock_analysis(
    prices: pd.DataFrame,
    strategies: dict[str, BaseStrategy],
    signals_by_strategy: dict[str, pd.DataFrame],
    *,
    backtest_config: BacktestConfig,
    start_date: date,
    end_date: date,
    expected_stock_count: int,
    top_fraction: float,
) -> dict[str, Any]:
    """Run each selected strategy independently on each existing stock."""
    symbols = sorted(map(str, prices["symbol"].dropna().unique()))
    if len(symbols) != expected_stock_count:
        raise ValueError(
            f"expected {expected_stock_count} stocks, found {len(symbols)}"
        )
    config = evaluation_config(
        backtest_config, start_date, end_date, split_name="analysis", max_positions=1
    )
    output: dict[str, Any] = {}
    for name in sorted(strategies):
        rows: list[dict[str, Any]] = []
        for symbol in symbols:
            result = run_backtest(
                signals_by_strategy[name].loc[
                    signals_by_strategy[name]["symbol"].eq(symbol)
                ],
                prices.loc[prices["symbol"].eq(symbol)],
                strategy=name,
                split="analysis",
                config=config,
            )
            rows.append({"symbol": symbol, "metrics": dict(result.metrics)})
        output[name] = {
            "label": "cross-sectional breadth analysis; not out-of-sample generalization",
            "stock_results": rows,
            "aggregate": aggregate_stock_results(rows, top_fraction=top_fraction),
        }
    return output


def build_regime_calendar(features: pd.DataFrame) -> pd.DataFrame:
    """Create one verified market/volatility regime row per feature date."""
    required = {"date", "market_trend_regime", "volatility_regime"}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"features are missing regime columns: {sorted(missing)}")
    frame = features[list(required)].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    non_null = frame.dropna(subset=["market_trend_regime", "volatility_regime"])
    conflicts = non_null.groupby("date")[["market_trend_regime", "volatility_regime"]].nunique()
    if (conflicts > 1).any(axis=None):
        raise ValueError("feature rows disagree on the regime for the same date")
    return (
        non_null.drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )


def assign_previous_regime(
    dates: pd.Series, regime_calendar: pd.DataFrame
) -> pd.DataFrame:
    """Assign only the regime from the strictly previous available trading day."""
    left = pd.DataFrame(
        {
            "_order": range(len(dates)),
            "date": pd.to_datetime(dates, errors="raise").dt.normalize(),
        }
    ).sort_values("date")
    assigned = pd.merge_asof(
        left,
        regime_calendar.sort_values("date"),
        on="date",
        direction="backward",
        allow_exact_matches=False,
    )
    return assigned.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def _regime_metrics(
    daily_returns: pd.Series,
    trades: pd.DataFrame,
    thresholds: RegimeThresholds,
    annualization_days: int,
    risk_free_rate: float,
) -> dict[str, Any]:
    clean_returns = pd.to_numeric(daily_returns, errors="coerce").dropna()
    observation_count = len(clean_returns)
    trade_count = len(trades)
    observation_ok = observation_count >= thresholds.minimum_observations
    trade_ok = trade_count >= thresholds.minimum_trades
    if observation_ok:
        cumulative = (1 + clean_returns).cumprod()
        equity = pd.concat([pd.Series([1.0]), cumulative], ignore_index=True)
        return_value = float(cumulative.iloc[-1] - 1) if not cumulative.empty else 0.0
        sharpe = calculate_sharpe(
            clean_returns,
            annualization_days=annualization_days,
            risk_free_rate=risk_free_rate,
        )
        drawdown = calculate_max_drawdown(equity)
    else:
        return_value = sharpe = drawdown = None
    win_rate = (
        float(pd.to_numeric(trades["net_pnl"]).gt(0).mean())
        if trade_ok and "net_pnl" in trades
        else None
    )
    return {
        "status": "RELIABLE" if observation_ok and trade_ok else "INSUFFICIENT_EVIDENCE",
        "number_of_observations": observation_count,
        "number_of_trades": trade_count,
        "return": return_value,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
        "win_rate": win_rate,
        "observation_threshold_met": observation_ok,
        "trade_threshold_met": trade_ok,
    }


def analyze_result_by_regime(
    result: BacktestResult,
    regime_calendar: pd.DataFrame,
    *,
    thresholds: RegimeThresholds,
    annualization_days: int,
    risk_free_rate: float,
) -> tuple[dict[str, Any], list[str]]:
    """Attribute returns and completed trades to prior-known regimes."""
    curve = result.equity_curve.copy()
    curve_regimes = assign_previous_regime(curve["date"], regime_calendar)
    curve[["market_trend_regime", "volatility_regime"]] = curve_regimes[
        ["market_trend_regime", "volatility_regime"]
    ]
    trades = result.trades.copy()
    if trades.empty:
        trades["market_trend_regime"] = pd.Series(dtype="string")
        trades["volatility_regime"] = pd.Series(dtype="string")
    else:
        trade_regimes = assign_previous_regime(trades["entry_date"], regime_calendar)
        trades[["market_trend_regime", "volatility_regime"]] = trade_regimes[
            ["market_trend_regime", "volatility_regime"]
        ]
    output: dict[str, Any] = {}
    warnings: list[str] = []
    for market in thresholds.market_values:
        for volatility in thresholds.volatility_values:
            key = f"{market}_{volatility}"
            daily_mask = curve["market_trend_regime"].eq(market) & curve[
                "volatility_regime"
            ].eq(volatility)
            trade_mask = trades["market_trend_regime"].eq(market) & trades[
                "volatility_regime"
            ].eq(volatility)
            metrics = _regime_metrics(
                curve.loc[daily_mask, "return"],
                trades.loc[trade_mask],
                thresholds,
                annualization_days,
                risk_free_rate,
            )
            output[key] = metrics
            if metrics["status"] == "INSUFFICIENT_EVIDENCE":
                warnings.append(f"{result.strategy}:{key}:INSUFFICIENT_EVIDENCE")
    unassigned_observations = int(
        curve[["market_trend_regime", "volatility_regime"]].isna().any(axis=1).sum()
    )
    unassigned_trades = int(
        trades[["market_trend_regime", "volatility_regime"]].isna().any(axis=1).sum()
    )
    output["assignment_diagnostics"] = {
        "unassigned_observations": unassigned_observations,
        "unassigned_trades": unassigned_trades,
    }
    return output, warnings


def run_regime_analysis(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    strategies: dict[str, BaseStrategy],
    signals_by_strategy: dict[str, pd.DataFrame],
    *,
    backtest_config: BacktestConfig,
    start_date: date,
    end_date: date,
    thresholds: RegimeThresholds,
) -> tuple[dict[str, Any], list[str]]:
    """Run portfolio backtests and describe results by prior-known regime."""
    calendar = build_regime_calendar(features)
    config = evaluation_config(backtest_config, start_date, end_date, split_name="analysis")
    output: dict[str, Any] = {}
    warnings: list[str] = []
    for name in sorted(strategies):
        result = run_backtest(
            signals_by_strategy[name],
            prices,
            strategy=name,
            split="analysis",
            config=config,
        )
        output[name], strategy_warnings = analyze_result_by_regime(
            result,
            calendar,
            thresholds=thresholds,
            annualization_days=backtest_config.annualization_days,
            risk_free_rate=backtest_config.risk_free_rate,
        )
        warnings.extend(strategy_warnings)
    return output, warnings


def assess_hypothesis(
    temporal_results: list[dict[str, Any]],
    sensitivity: dict[str, Any],
    cross_stock: dict[str, Any],
    regimes: dict[str, Any],
    thresholds: HypothesisThresholds,
) -> dict[str, Any]:
    """Create a deterministic descriptive summary without selecting strategies."""
    fold_winners: dict[str, str] = {}
    for fold in sorted({row["fold"] for row in temporal_results}):
        rows = [row for row in temporal_results if row["fold"] == fold]
        if rows:
            winner = sorted(
                rows,
                key=lambda row: (
                    -float(row["metrics"]["cumulative_return"]),
                    row["strategy"],
                ),
            )[0]
            fold_winners[fold] = winner["strategy"]
    winner_set = set(fold_winners.values())
    momentum_dominant = bool(fold_winners) and winner_set == {"momentum"}
    strategy_winning_periods = {
        name: [fold for fold, winner in fold_winners.items() if winner == name]
        for name in sorted(sensitivity)
    }

    regime_winners: dict[str, str] = {}
    regime_spreads: dict[str, float] = {}
    regime_keys = sorted(
        {
            key
            for strategy_results in regimes.values()
            for key in strategy_results
            if key != "assignment_diagnostics"
        }
    )
    for key in regime_keys:
        comparable = [
            (name, values[key]["sharpe"])
            for name, values in regimes.items()
            if values.get(key, {}).get("status") == "RELIABLE"
            and values[key].get("sharpe") is not None
        ]
        if len(comparable) >= 2:
            comparable.sort(key=lambda item: (-float(item[1]), item[0]))
            spread = float(max(item[1] for item in comparable) - min(item[1] for item in comparable))
            regime_spreads[key] = spread
            if spread >= thresholds.regime_sharpe_spread:
                regime_winners[key] = comparable[0][0]
    differentiated_regimes = len(set(regime_winners.values())) > 1
    different_period_winners = len(winner_set) > 1

    concentrations = {
        name: values["aggregate"]["positive_return_concentration_top_fraction"]
        for name, values in cross_stock.items()
    }
    valid_concentrations = [value for value in concentrations.values() if value is not None]
    median_concentration = (
        float(pd.Series(valid_concentrations).median()) if valid_concentrations else None
    )
    concentrated = (
        median_concentration is not None
        and median_concentration >= thresholds.stock_concentration_threshold
    )
    sensitivity_counts = {
        classification: sum(
            values["classification"] == classification for values in sensitivity.values()
        )
        for classification in sorted(CLASSIFICATIONS)
    }

    if not fold_winners or len(regime_winners) < 2:
        conclusion = "INSUFFICIENT_EVIDENCE"
    elif different_period_winners and differentiated_regimes:
        conclusion = "SUPPORTIVE"
    elif different_period_winners or differentiated_regimes:
        conclusion = "MIXED"
    else:
        conclusion = "WEAK"
    return {
        "conclusion": conclusion,
        "hypothesis": "Different market contexts can favor different strategy families.",
        "momentum_dominant_across_all_periods": momentum_dominant,
        "temporal_winners": fold_winners,
        "strategy_winning_periods": strategy_winning_periods,
        "different_strategies_win_different_periods": different_period_winners,
        "regime_winners_with_material_sharpe_spread": regime_winners,
        "regime_sharpe_spreads": regime_spreads,
        "different_strategies_win_different_regimes": differentiated_regimes,
        "performance_concentrated_in_small_number_of_stocks": concentrated,
        "stock_positive_return_concentration": concentrations,
        "median_positive_return_concentration": median_concentration,
        "parameter_sensitivity_classification_counts": sensitivity_counts,
        "explanation": (
            "The conclusion uses only differences in yearly return leaders and "
            "reliable regime Sharpe leaders. It is descriptive and does not alter "
            "strategy parameters or optimization results."
        ),
    }


def run_robustness_analysis(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    optimization_report: dict[str, Any],
    *,
    backtest_config: BacktestConfig,
    optimization_config: OptimizationConfig,
    robustness_config: RobustnessConfig,
) -> dict[str, Any]:
    """Run the complete read-only robustness milestone on pre-TEST data."""
    feature_dates = pd.to_datetime(features["date"], errors="raise")
    if (feature_dates.dt.date >= robustness_config.test_cutoff).any():
        raise ValueError("features include frozen 2026 TEST rows")
    price_column = "timestamp" if "timestamp" in prices else "date"
    price_dates = (
        pd.to_datetime(prices[price_column], utc=True, errors="raise")
        .dt.tz_convert("Asia/Kolkata")
        .dt.date
    )
    if (price_dates >= robustness_config.test_cutoff).any():
        raise ValueError("prices include frozen 2026 TEST rows")

    strategies = load_selected_strategies(optimization_report)
    signals = {
        name: strategy.generate_signals(features)
        for name, strategy in sorted(strategies.items())
    }
    temporal = run_temporal_stability(
        features,
        prices,
        strategies,
        robustness_config.temporal_folds,
        backtest_config=backtest_config,
        test_cutoff=robustness_config.test_cutoff,
        signals_by_strategy=signals,
    )
    sensitivity = run_parameter_sensitivity(
        features,
        prices,
        strategies,
        optimization_report,
        backtest_config=backtest_config,
        optimization_config=optimization_config,
        thresholds=robustness_config.sensitivity,
    )
    cross_stock = run_cross_stock_analysis(
        prices,
        strategies,
        signals,
        backtest_config=backtest_config,
        start_date=robustness_config.analysis_start_date,
        end_date=robustness_config.analysis_end_date,
        expected_stock_count=robustness_config.expected_stock_count,
        top_fraction=robustness_config.hypothesis.stock_concentration_top_fraction,
    )
    regimes, warnings = run_regime_analysis(
        features,
        prices,
        strategies,
        signals,
        backtest_config=backtest_config,
        start_date=robustness_config.analysis_start_date,
        end_date=robustness_config.analysis_end_date,
        thresholds=robustness_config.regime,
    )
    hypothesis = assess_hypothesis(
        temporal,
        sensitivity,
        cross_stock,
        regimes,
        robustness_config.hypothesis,
    )
    return {
        "methodology": {
            "analysis_type": "robustness_and_diagnostic_analysis",
            "temporal_interpretation": (
                "Fixed-parameter temporal stability; not new unseen out-of-sample validation."
            ),
            "parameters": "already selected TRAIN 2020-2024 / VALIDATION 2025 configurations",
            "optimization_performed": False,
            "test_period_accessed": False,
            "test_cutoff": robustness_config.test_cutoff.isoformat(),
            "execution": "existing next-open engine with existing costs and slippage",
            "cross_stock_interpretation": (
                "cross-sectional breadth analysis; not out-of-sample generalization"
            ),
            "regime_attribution": (
                "strictly previous available trading-day regime for returns and entry trades"
            ),
        },
        "date_boundaries": {
            "analysis_start_date": robustness_config.analysis_start_date.isoformat(),
            "analysis_end_date": robustness_config.analysis_end_date.isoformat(),
            "temporal_folds": [
                {
                    "name": fold.name,
                    "context_start_date": fold.context_start_date.isoformat(),
                    "context_end_date": fold.context_end_date.isoformat(),
                    "evaluation_start_date": fold.evaluation_start_date.isoformat(),
                    "evaluation_end_date": fold.evaluation_end_date.isoformat(),
                }
                for fold in robustness_config.temporal_folds
            ],
        },
        "temporal_results": temporal,
        "parameter_sensitivity": sensitivity,
        "stock_level_results": {
            name: values["stock_results"] for name, values in cross_stock.items()
        },
        "aggregate_stock_breadth": {
            name: {
                "label": values["label"],
                **values["aggregate"],
            }
            for name, values in cross_stock.items()
        },
        "regime_results": regimes,
        "sample_size_warnings": sorted(warnings),
        "hypothesis_assessment": hypothesis,
    }

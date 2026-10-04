"""Train/validation-only deterministic strategy parameter optimization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.backtesting.engine import BacktestConfig, run_backtest
from src.evaluation.sensitivity import run_sensitivity_analysis
from src.evaluation.validation import (
    calculate_objective,
    rank_candidates,
    validate_strategy_result,
)
from src.strategies.base import BaseStrategy
from src.strategies.registry import STRATEGY_TYPES, load_enabled_strategies


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "optimization.yaml"


@dataclass(frozen=True)
class OptimizationConfig:
    """Small-grid search and risk-adjusted selection settings."""

    sharpe_weight: float
    return_weight: float
    drawdown_weight: float
    top_train_candidates: int
    minimum_trade_count_warning: int
    output_path: Path
    parameter_grids: dict[str, dict[str, list[Any]]]


def load_optimization_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> OptimizationConfig:
    """Load the deterministic optimization configuration."""
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = (yaml.safe_load(handle) or {}).get("optimization", {})
    objective = raw.get("objective") or {}
    config = OptimizationConfig(
        sharpe_weight=float(objective["sharpe_weight"]),
        return_weight=float(objective["return_weight"]),
        drawdown_weight=float(objective["drawdown_weight"]),
        top_train_candidates=int(raw["top_train_candidates"]),
        minimum_trade_count_warning=int(raw["minimum_trade_count_warning"]),
        output_path=PROJECT_ROOT / str(raw["output_path"]),
        parameter_grids=raw["parameter_grids"],
    )
    if config.top_train_candidates <= 0 or config.minimum_trade_count_warning < 0:
        raise ValueError("optimization candidate/trade settings are invalid")
    if min(config.sharpe_weight, config.return_weight, config.drawdown_weight) < 0:
        raise ValueError("objective weights cannot be negative")
    if not isinstance(config.parameter_grids, dict):
        raise ValueError("parameter_grids must be a mapping")
    return config


def _candidate_id(strategy: str, params: dict[str, Any]) -> str:
    payload = strategy + "|" + json.dumps(
        params, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _evaluate_candidate(
    strategy_name: str,
    params: dict[str, Any],
    split: str,
    features: pd.DataFrame,
    prices: pd.DataFrame,
    backtest_config: BacktestConfig,
    optimization_config: OptimizationConfig,
    baseline_params: dict[str, Any],
) -> dict[str, Any]:
    strategy = STRATEGY_TYPES[strategy_name](params)
    signals = strategy.generate_signals(features)
    result = run_backtest(
        signals,
        prices,
        strategy=strategy_name,
        split=split,
        config=backtest_config,
    )
    objective = calculate_objective(
        result.metrics,
        sharpe_weight=optimization_config.sharpe_weight,
        return_weight=optimization_config.return_weight,
        drawdown_weight=optimization_config.drawdown_weight,
    )
    record = {
        "strategy": strategy_name,
        "candidate_id": _candidate_id(strategy_name, params),
        "params": params,
        "is_baseline": params == baseline_params,
        "split": split,
        "objective": objective,
        **result.metrics,
    }
    validate_strategy_result(record)
    return record


def _overfitting_flags(
    selected_train: dict[str, Any],
    selected_validation: dict[str, Any],
    grid: dict[str, list[Any]],
    minimum_trade_count: int,
) -> list[str]:
    flags: list[str] = []
    if selected_validation["objective"] < 0 <= selected_train["objective"]:
        flags.append("positive_train_but_negative_validation_objective")
    if selected_validation["objective"] < selected_train["objective"] - 1.0:
        flags.append("large_train_to_validation_objective_drop")
    if selected_validation["number_of_trades"] < minimum_trade_count:
        flags.append("few_validation_trades")
    for name, values in sorted(grid.items()):
        numeric = [value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
        selected_value = selected_train["params"].get(name)
        if len(set(numeric)) > 2 and selected_value in {min(numeric), max(numeric)}:
            flags.append(f"selected_{name}_at_grid_boundary")
    return flags


def optimize_strategies(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    backtest_config: BacktestConfig,
    optimization_config: OptimizationConfig,
    strategies: list[BaseStrategy] | None = None,
) -> dict[str, Any]:
    """Search TRAIN, compare shortlisted candidates on VALIDATION, never TEST."""
    enabled = strategies if strategies is not None else load_enabled_strategies()
    all_records: list[dict[str, Any]] = []
    selections: dict[str, Any] = {}

    for baseline_strategy in enabled:
        name = baseline_strategy.name
        baseline = dict(baseline_strategy.params)
        grid = optimization_config.parameter_grids.get(name)
        if not isinstance(grid, dict):
            raise ValueError(f"parameter grid is missing for {name}")
        unknown = set(grid).difference(baseline)
        if unknown:
            raise ValueError(f"{name} grid contains unsupported parameters: {sorted(unknown)}")
        candidates = run_sensitivity_analysis(baseline, grid)
        train_records = [
            _evaluate_candidate(
                name,
                params,
                "train",
                features,
                prices,
                backtest_config,
                optimization_config,
                baseline,
            )
            for params in candidates
        ]
        ranked_train = rank_candidates(train_records)
        all_records.extend(train_records)

        shortlisted_ids = {
            record["candidate_id"]
            for record in ranked_train[: optimization_config.top_train_candidates]
        }
        baseline_id = _candidate_id(name, baseline)
        shortlisted_ids.add(baseline_id)
        params_by_id = {
            _candidate_id(name, params): params for params in candidates
        }
        validation_records = [
            _evaluate_candidate(
                name,
                params_by_id[candidate_id],
                "validation",
                features,
                prices,
                backtest_config,
                optimization_config,
                baseline,
            )
            for candidate_id in sorted(shortlisted_ids)
        ]
        all_records.extend(validation_records)
        train_by_id = {record["candidate_id"]: record for record in train_records}
        validation_by_id = {
            record["candidate_id"]: record for record in validation_records
        }
        combined = [
            {
                "candidate_id": candidate_id,
                "params": params_by_id[candidate_id],
                "combined_objective": (
                    train_by_id[candidate_id]["objective"]
                    + validation_by_id[candidate_id]["objective"]
                )
                / 2,
                "validation_objective": validation_by_id[candidate_id]["objective"],
                "train_objective": train_by_id[candidate_id]["objective"],
            }
            for candidate_id in shortlisted_ids
        ]
        selected = sorted(
            combined,
            key=lambda item: (
                -item["combined_objective"],
                -item["validation_objective"],
                -item["train_objective"],
                json.dumps(item["params"], sort_keys=True, separators=(",", ":")),
            ),
        )[0]
        selected_id = selected["candidate_id"]
        selections[name] = {
            "candidate_count": len(candidates),
            "baseline_params": baseline,
            "baseline_train": train_by_id[baseline_id],
            "baseline_validation": validation_by_id[baseline_id],
            "best_train": ranked_train[0],
            "selected_params": selected["params"],
            "selected_train": train_by_id[selected_id],
            "selected_validation": validation_by_id[selected_id],
            "combined_objective": selected["combined_objective"],
            "overfitting_flags": _overfitting_flags(
                train_by_id[selected_id],
                validation_by_id[selected_id],
                grid,
                optimization_config.minimum_trade_count_warning,
            ),
        }

    return {
        "methodology": {
            "search_split": "train",
            "comparison_split": "validation",
            "excluded_split": "test",
            "top_train_candidates": optimization_config.top_train_candidates,
            "objective": (
                f"{optimization_config.sharpe_weight}*sharpe + "
                f"{optimization_config.return_weight}*cumulative_return - "
                f"{optimization_config.drawdown_weight}*max_drawdown"
            ),
            "selection": "highest equal-weight mean of TRAIN and VALIDATION objective",
        },
        "candidate_results": sorted(
            all_records,
            key=lambda record: (
                record["strategy"], record["split"], record["candidate_id"]
            ),
        ),
        "selections": selections,
    }

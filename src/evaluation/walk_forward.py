"""Fixed-parameter temporal-stability utilities.

These helpers deliberately do not optimize parameters.  The selected strategy
configuration has already seen the requested years through the original
TRAIN/VALIDATION process, so this is diagnostic stability analysis rather than
new out-of-sample validation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

import pandas as pd

from src.backtesting.engine import BacktestConfig, SplitBoundary, run_backtest
from src.strategies.base import BaseStrategy


@dataclass(frozen=True)
class TemporalFold:
    """One context range and its non-overlapping evaluation year."""

    name: str
    context_start_date: date
    context_end_date: date
    evaluation_start_date: date
    evaluation_end_date: date


def validate_temporal_folds(folds: tuple[TemporalFold, ...], test_cutoff: date) -> None:
    """Reject overlapping evaluation periods or any access to TEST."""
    if not folds:
        raise ValueError("at least one temporal fold is required")
    ordered = sorted(folds, key=lambda fold: fold.evaluation_start_date)
    previous_end: date | None = None
    for fold in ordered:
        if fold.context_start_date > fold.context_end_date:
            raise ValueError(f"{fold.name} context ends before it starts")
        if fold.context_end_date >= fold.evaluation_start_date:
            raise ValueError(f"{fold.name} context overlaps its evaluation period")
        if fold.evaluation_start_date > fold.evaluation_end_date:
            raise ValueError(f"{fold.name} evaluation ends before it starts")
        if fold.evaluation_end_date >= test_cutoff:
            raise ValueError(f"{fold.name} accesses the frozen TEST period")
        if previous_end is not None and fold.evaluation_start_date <= previous_end:
            raise ValueError("temporal evaluation periods overlap")
        previous_end = fold.evaluation_end_date


def evaluation_config(
    base: BacktestConfig,
    start_date: date,
    end_date: date,
    *,
    split_name: str = "evaluation",
    max_positions: int | None = None,
) -> BacktestConfig:
    """Reuse the engine with an isolated in-memory diagnostic boundary."""
    return replace(
        base,
        max_positions=base.max_positions if max_positions is None else max_positions,
        splits={split_name: SplitBoundary(start_date, end_date)},
    )


def run_temporal_stability(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    strategies: dict[str, BaseStrategy],
    folds: tuple[TemporalFold, ...],
    *,
    backtest_config: BacktestConfig,
    test_cutoff: date,
    signals_by_strategy: dict[str, pd.DataFrame] | None = None,
) -> list[dict]:
    """Evaluate fixed selected parameters in fresh, isolated yearly periods."""
    validate_temporal_folds(folds, test_cutoff)
    cached = signals_by_strategy or {
        name: strategy.generate_signals(features)
        for name, strategy in sorted(strategies.items())
    }
    rows: list[dict] = []
    for fold in folds:
        config = evaluation_config(
            backtest_config,
            fold.evaluation_start_date,
            fold.evaluation_end_date,
        )
        for name in sorted(strategies):
            result = run_backtest(
                cached[name],
                prices,
                strategy=name,
                split="evaluation",
                config=config,
            )
            rows.append(
                {
                    "fold": fold.name,
                    "context_start_date": fold.context_start_date.isoformat(),
                    "context_end_date": fold.context_end_date.isoformat(),
                    "evaluation_start_date": fold.evaluation_start_date.isoformat(),
                    "evaluation_end_date": fold.evaluation_end_date.isoformat(),
                    "strategy": name,
                    "metrics": dict(result.metrics),
                    "fresh_initial_cash": backtest_config.initial_cash,
                    "ending_position": (
                        "flat"
                        if result.equity_curve.empty
                        else str(result.equity_curve.iloc[-1]["position"])
                    ),
                }
            )
    return rows

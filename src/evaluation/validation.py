"""Risk-adjusted candidate scoring and deterministic ranking."""

from __future__ import annotations

import json
from typing import Any


def calculate_objective(
    metrics: dict[str, float | int],
    *,
    sharpe_weight: float,
    return_weight: float,
    drawdown_weight: float,
) -> float:
    """Score return and Sharpe while explicitly penalizing drawdown."""
    return float(
        sharpe_weight * float(metrics["sharpe"])
        + return_weight * float(metrics["cumulative_return"])
        - drawdown_weight * float(metrics["max_drawdown"])
    )


def rank_candidates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank candidates deterministically with documented tie breakers."""
    return sorted(
        records,
        key=lambda record: (
            -float(record["objective"]),
            -float(record["sharpe"]),
            -float(record["cumulative_return"]),
            float(record["max_drawdown"]),
            -int(record["number_of_trades"]),
            json.dumps(record["params"], sort_keys=True, separators=(",", ":")),
        ),
    )


def validate_strategy_result(record: dict[str, Any]) -> None:
    """Reject incomplete candidate records."""
    required = {
        "strategy",
        "candidate_id",
        "params",
        "split",
        "objective",
        "cumulative_return",
        "sharpe",
        "max_drawdown",
        "volatility",
        "number_of_trades",
        "win_rate",
    }
    missing = required.difference(record)
    if missing:
        raise ValueError(f"candidate result is missing fields: {sorted(missing)}")

"""Deterministic, compact parameter-grid generation."""

from __future__ import annotations

from itertools import product
from typing import Any


def run_sensitivity_analysis(
    baseline_params: dict[str, Any],
    parameter_grid: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    """Return the deterministic Cartesian grid merged with fixed baseline values."""
    if not parameter_grid:
        return [dict(baseline_params)]
    keys = sorted(parameter_grid)
    values = []
    for key in keys:
        choices = parameter_grid[key]
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"parameter grid for {key} must be a non-empty list")
        values.append(choices)
    candidates: list[dict[str, Any]] = []
    for combination in product(*values):
        params = dict(baseline_params)
        params.update(dict(zip(keys, combination)))
        if params not in candidates:
            candidates.append(params)
    if baseline_params not in candidates:
        candidates.append(dict(baseline_params))
    return candidates

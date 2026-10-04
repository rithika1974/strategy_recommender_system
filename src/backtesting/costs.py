"""Deterministic proportional transaction-cost model."""

from __future__ import annotations


def apply_costs(notional: float, transaction_cost_bps: float) -> float:
    """Return the non-negative cost charged on an absolute trade notional."""
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps cannot be negative")
    return abs(float(notional)) * float(transaction_cost_bps) / 10_000

"""Lookahead-safe deterministic fill-price simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.backtesting.costs import apply_costs


OrderSide = Literal["buy", "sell"]


@dataclass(frozen=True)
class Execution:
    """One simulated fill."""

    side: OrderSide
    quantity: float
    reference_price: float
    execution_price: float
    transaction_cost: float


def simulate_execution(
    reference_price: float,
    quantity: float,
    side: OrderSide,
    *,
    slippage_bps: float,
    transaction_cost_bps: float,
) -> Execution:
    """Apply adverse slippage and proportional costs to a buy or sell."""
    if reference_price <= 0 or quantity <= 0:
        raise ValueError("reference_price and quantity must be positive")
    if side not in {"buy", "sell"}:
        raise ValueError(f"unsupported order side: {side}")
    if slippage_bps < 0:
        raise ValueError("slippage_bps cannot be negative")
    direction = 1 if side == "buy" else -1
    execution_price = float(reference_price) * (
        1 + direction * float(slippage_bps) / 10_000
    )
    cost = apply_costs(execution_price * quantity, transaction_cost_bps)
    return Execution(
        side=side,
        quantity=float(quantity),
        reference_price=float(reference_price),
        execution_price=execution_price,
        transaction_cost=cost,
    )

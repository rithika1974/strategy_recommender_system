"""Portfolio and trade-ledger state used by the backtesting engine."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.backtesting.execution import Execution


@dataclass
class Position:
    """One open long or short position."""

    symbol: str
    strategy: str
    direction: str
    quantity: float
    entry_date: str
    entry_price: float
    entry_cost: float


@dataclass
class PortfolioState:
    """Mutable cash, positions, realized P&L, and completed trades."""

    initial_cash: float
    cash: float = field(init=False)
    positions: dict[str, Position] = field(default_factory=dict)
    trades: list[dict] = field(default_factory=list)
    realized_pnl: float = 0.0

    def __post_init__(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        self.cash = float(self.initial_cash)

    def equity(self, mark_prices: dict[str, float]) -> float:
        value = self.cash
        for symbol, position in self.positions.items():
            price = mark_prices.get(symbol, position.entry_price)
            signed_quantity = (
                position.quantity if position.direction == "long" else -position.quantity
            )
            value += signed_quantity * price
        return float(value)

    def unrealized_pnl(self, mark_prices: dict[str, float]) -> float:
        total = 0.0
        for symbol, position in self.positions.items():
            price = mark_prices.get(symbol, position.entry_price)
            multiplier = 1 if position.direction == "long" else -1
            total += multiplier * (price - position.entry_price) * position.quantity
        return float(total)

    def open_position(
        self,
        symbol: str,
        strategy: str,
        direction: str,
        date: str,
        execution: Execution,
    ) -> None:
        if symbol in self.positions or direction not in {"long", "short"}:
            raise ValueError("position open request is invalid")
        notional = execution.execution_price * execution.quantity
        if direction == "long":
            self.cash -= notional + execution.transaction_cost
        else:
            self.cash += notional - execution.transaction_cost
        self.positions[symbol] = Position(
            symbol=symbol,
            strategy=strategy,
            direction=direction,
            quantity=execution.quantity,
            entry_date=date,
            entry_price=execution.execution_price,
            entry_cost=execution.transaction_cost,
        )

    def close_position(
        self,
        symbol: str,
        date: str,
        execution: Execution,
    ) -> dict:
        position = self.positions.pop(symbol)
        notional = execution.execution_price * execution.quantity
        if position.direction == "long":
            self.cash += notional - execution.transaction_cost
            gross_pnl = (
                execution.execution_price - position.entry_price
            ) * position.quantity
        else:
            self.cash -= notional + execution.transaction_cost
            gross_pnl = (
                position.entry_price - execution.execution_price
            ) * position.quantity
        costs = position.entry_cost + execution.transaction_cost
        net_pnl = gross_pnl - costs
        self.realized_pnl += net_pnl
        trade = {
            "symbol": position.symbol,
            "strategy": position.strategy,
            "entry_date": position.entry_date,
            "entry_price": position.entry_price,
            "exit_date": date,
            "exit_price": execution.execution_price,
            "direction": position.direction,
            "quantity": position.quantity,
            "gross_pnl": gross_pnl,
            "costs": costs,
            "net_pnl": net_pnl,
        }
        self.trades.append(trade)
        return trade


def build_portfolio(initial_cash: float) -> PortfolioState:
    """Create an empty deterministic portfolio."""
    return PortfolioState(initial_cash=initial_cash)

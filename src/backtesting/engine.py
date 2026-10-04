"""Deterministic multi-symbol historical backtesting engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.backtesting.execution import simulate_execution
from src.backtesting.portfolio import PortfolioState, build_portfolio
from src.evaluation.metrics import compute_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "backtest.yaml"
TRADE_COLUMNS = [
    "symbol",
    "strategy",
    "entry_date",
    "entry_price",
    "exit_date",
    "exit_price",
    "direction",
    "quantity",
    "gross_pnl",
    "costs",
    "net_pnl",
]


@dataclass(frozen=True)
class SplitBoundary:
    """Inclusive dates for one isolated chronological split."""

    start_date: date
    end_date: date | None


@dataclass(frozen=True)
class BacktestConfig:
    """Validated deterministic portfolio and execution settings."""

    initial_cash: float
    position_size: float
    transaction_cost_bps: float
    slippage_bps: float
    max_positions: int
    annualization_days: int
    risk_free_rate: float
    splits: dict[str, SplitBoundary]


@dataclass
class BacktestResult:
    """Equity, completed trades, metrics, and input diagnostics."""

    strategy: str
    split: str
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float | int]
    diagnostics: dict[str, int]


def _as_date(value: Any, name: str, *, optional: bool = False) -> date | None:
    if value is None and optional:
        return None
    if value is None:
        raise ValueError(f"{name} must be configured")
    try:
        return pd.Timestamp(value).date()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid date") from exc


def load_backtest_config(path: str | Path = DEFAULT_CONFIG_PATH) -> BacktestConfig:
    """Load backtest settings from the existing YAML configuration."""
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = (yaml.safe_load(handle) or {}).get("backtest", {})
    if raw.get("execution") != "next_open":
        raise ValueError("V1 backtesting supports only next_open execution")
    split_entries = raw.get("splits")
    if not isinstance(split_entries, dict):
        raise ValueError("backtest.splits must be configured")
    splits: dict[str, SplitBoundary] = {}
    for name in ("train", "validation", "test"):
        entry = split_entries.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"backtest split is missing: {name}")
        start = _as_date(entry.get("start_date"), f"{name}.start_date")
        end = _as_date(entry.get("end_date"), f"{name}.end_date", optional=True)
        if end is not None and end < start:
            raise ValueError(f"{name} split ends before it starts")
        splits[name] = SplitBoundary(start, end)
    if splits["train"].end_date is None or splits["validation"].end_date is None:
        raise ValueError("train and validation require explicit end dates")
    if splits["train"].end_date >= splits["validation"].start_date:
        raise ValueError("train and validation splits overlap")
    if splits["validation"].end_date >= splits["test"].start_date:
        raise ValueError("validation and test splits overlap")
    config = BacktestConfig(
        initial_cash=float(raw["initial_cash"]),
        position_size=float(raw["position_size"]),
        transaction_cost_bps=float(raw["transaction_cost_bps"]),
        slippage_bps=float(raw["slippage_bps"]),
        max_positions=int(raw["max_positions"]),
        annualization_days=int(raw["annualization_days"]),
        risk_free_rate=float(raw["risk_free_rate"]),
        splits=splits,
    )
    if config.initial_cash <= 0:
        raise ValueError("initial_cash must be positive")
    if not 0 < config.position_size <= 1:
        raise ValueError("position_size must be in (0, 1]")
    if config.max_positions <= 0 or config.annualization_days <= 0:
        raise ValueError("max_positions and annualization_days must be positive")
    if config.transaction_cost_bps < 0 or config.slippage_bps < 0:
        raise ValueError("cost and slippage settings cannot be negative")
    return config


def _within_split(frame: pd.DataFrame, boundary: SplitBoundary) -> pd.DataFrame:
    mask = frame["_date"].dt.date >= boundary.start_date
    if boundary.end_date is not None:
        mask &= frame["_date"].dt.date <= boundary.end_date
    return frame.loc[mask].copy()


def _prepare_prices(
    prices: pd.DataFrame, boundary: SplitBoundary
) -> tuple[pd.DataFrame, int]:
    required = {"symbol", "open", "close"}
    date_column = "timestamp" if "timestamp" in prices else "date"
    missing = required.difference(prices.columns)
    if missing or date_column not in prices:
        raise ValueError(f"prices are missing columns: {sorted(missing | {date_column} - set(prices.columns))}")
    frame = prices[["symbol", date_column, "open", "close"]].copy()
    frame["_date"] = (
        pd.to_datetime(frame[date_column], utc=True, errors="raise")
        .dt.tz_convert("Asia/Kolkata")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    frame = _within_split(frame, boundary)
    invalid = frame[["open", "close"]].isna().any(axis=1) | frame["open"].le(0) | frame["close"].le(0)
    skipped = int(invalid.sum())
    frame = frame.loc[~invalid].copy()
    frame["date"] = frame["_date"].dt.strftime("%Y-%m-%d")
    if frame.duplicated(["symbol", "date"]).any():
        raise ValueError("prices contain duplicate (symbol, date) rows")
    return frame.sort_values(["date", "symbol"]).reset_index(drop=True), skipped


def _prepare_signals(
    signals: pd.DataFrame,
    strategy: str,
    boundary: SplitBoundary,
) -> tuple[pd.DataFrame, int]:
    required = {"symbol", "date", "signal"}
    missing = required.difference(signals.columns)
    if missing:
        raise ValueError(f"signals are missing columns: {sorted(missing)}")
    frame = signals.copy()
    if "strategy" in frame:
        frame = frame.loc[frame["strategy"] == strategy].copy()
    frame["_date"] = pd.to_datetime(frame["date"], utc=True, errors="raise").dt.tz_localize(None).dt.normalize()
    frame = _within_split(frame, boundary)
    if frame.duplicated(["symbol", "date"]).any():
        raise ValueError("signals contain duplicate (symbol, date) rows")
    null_signals = int(frame["signal"].isna().sum())
    frame["signal"] = frame["signal"].fillna("flat")
    invalid = set(frame["signal"].unique()).difference({"long", "short", "flat"})
    if invalid:
        raise ValueError(f"unsupported signals: {sorted(invalid)}")
    return frame.sort_values(["symbol", "_date"]).reset_index(drop=True), null_signals


def _next_open_targets(prices: pd.DataFrame, signals: pd.DataFrame) -> dict[tuple[str, str], str]:
    targets: dict[tuple[str, str], str] = {}
    for symbol, symbol_prices in prices.groupby("symbol", sort=True):
        symbol_signals = signals.loc[signals["symbol"] == symbol]
        if symbol_signals.empty:
            continue
        signal_dates = symbol_signals["_date"].to_numpy()
        signal_values = symbol_signals["signal"].to_numpy()
        for row in symbol_prices.itertuples(index=False):
            index = signal_dates.searchsorted(pd.Timestamp(row.date), side="left") - 1
            if index >= 0:
                targets[(row.date, str(symbol))] = str(signal_values[index])
    return targets


def _position_label(portfolio: PortfolioState) -> str:
    directions = {position.direction for position in portfolio.positions.values()}
    if not directions:
        return "flat"
    if len(directions) == 1:
        return next(iter(directions))
    return "mixed"


def _weighted_execution_price(executions: list[tuple[float, float]]) -> float | None:
    quantity = sum(item[1] for item in executions)
    if quantity == 0:
        return None
    return sum(price * size for price, size in executions) / quantity


def run_backtest(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    strategy: str,
    split: str,
    config: BacktestConfig,
) -> BacktestResult:
    """Simulate one strategy/split using close-t signals at the next open."""
    if split not in config.splits:
        raise ValueError(f"unknown split: {split}")
    price_frame, skipped_prices = _prepare_prices(prices, config.splits[split])
    signal_frame, null_signals = _prepare_signals(signals, strategy, config.splits[split])
    targets = _next_open_targets(price_frame, signal_frame)
    portfolio = build_portfolio(config.initial_cash)
    mark_prices: dict[str, float] = {}
    mark_dates: dict[str, str] = {}
    curve_rows: list[dict[str, Any]] = []

    for current_date, daily_prices in price_frame.groupby("date", sort=True):
        rows = {str(row.symbol): row for row in daily_prices.itertuples(index=False)}
        for symbol, row in rows.items():
            mark_prices[symbol] = float(row.open)
            mark_dates[symbol] = current_date
        executions: list[tuple[float, float]] = []
        daily_cost = 0.0

        for symbol in sorted(set(portfolio.positions).intersection(rows)):
            desired = targets.get((current_date, symbol), "flat")
            position = portfolio.positions[symbol]
            if desired == position.direction:
                continue
            side = "sell" if position.direction == "long" else "buy"
            execution = simulate_execution(
                float(rows[symbol].open),
                position.quantity,
                side,
                slippage_bps=config.slippage_bps,
                transaction_cost_bps=config.transaction_cost_bps,
            )
            portfolio.close_position(symbol, current_date, execution)
            daily_cost += execution.transaction_cost
            executions.append((execution.execution_price, execution.quantity))

        for symbol in sorted(rows):
            desired = targets.get((current_date, symbol), "flat")
            if desired == "flat" or symbol in portfolio.positions:
                continue
            if len(portfolio.positions) >= config.max_positions:
                break
            side = "buy" if desired == "long" else "sell"
            equity_at_open = portfolio.equity(mark_prices)
            target_notional = equity_at_open * config.position_size
            slipped_unit = simulate_execution(
                float(rows[symbol].open),
                1.0,
                side,
                slippage_bps=config.slippage_bps,
                transaction_cost_bps=config.transaction_cost_bps,
            )
            quantity = target_notional / (
                slipped_unit.execution_price
                * (1 + config.transaction_cost_bps / 10_000)
            )
            execution = simulate_execution(
                float(rows[symbol].open),
                quantity,
                side,
                slippage_bps=config.slippage_bps,
                transaction_cost_bps=config.transaction_cost_bps,
            )
            portfolio.open_position(symbol, strategy, desired, current_date, execution)
            daily_cost += execution.transaction_cost
            executions.append((execution.execution_price, execution.quantity))

        for symbol, row in rows.items():
            mark_prices[symbol] = float(row.close)
        curve_rows.append(
            {
                "date": current_date,
                "cash": portfolio.cash,
                "position": _position_label(portfolio),
                "quantity": sum(p.quantity for p in portfolio.positions.values()),
                "execution_price": _weighted_execution_price(executions),
                "transaction_cost": daily_cost,
                "realized_pnl": portfolio.realized_pnl,
                "unrealized_pnl": portfolio.unrealized_pnl(mark_prices),
                "equity": portfolio.equity(mark_prices),
            }
        )

    if curve_rows and portfolio.positions:
        final_executions: list[tuple[float, float]] = []
        final_cost = 0.0
        for symbol in sorted(list(portfolio.positions)):
            position = portfolio.positions[symbol]
            side = "sell" if position.direction == "long" else "buy"
            execution = simulate_execution(
                mark_prices[symbol],
                position.quantity,
                side,
                slippage_bps=config.slippage_bps,
                transaction_cost_bps=config.transaction_cost_bps,
            )
            portfolio.close_position(symbol, mark_dates[symbol], execution)
            final_cost += execution.transaction_cost
            final_executions.append((execution.execution_price, execution.quantity))
        final_row = curve_rows[-1]
        final_row.update(
            {
                "cash": portfolio.cash,
                "position": "flat",
                "quantity": 0.0,
                "execution_price": _weighted_execution_price(final_executions),
                "transaction_cost": final_row["transaction_cost"] + final_cost,
                "realized_pnl": portfolio.realized_pnl,
                "unrealized_pnl": 0.0,
                "equity": portfolio.cash,
            }
        )

    equity_curve = pd.DataFrame(curve_rows)
    equity_curve.attrs["initial_cash"] = config.initial_cash
    if not equity_curve.empty:
        equity_curve["return"] = equity_curve["equity"].pct_change().fillna(
            equity_curve["equity"] / config.initial_cash - 1
        )
    trades = pd.DataFrame(portfolio.trades, columns=TRADE_COLUMNS)
    metrics = compute_metrics(
        equity_curve,
        trades,
        annualization_days=config.annualization_days,
        risk_free_rate=config.risk_free_rate,
    )
    return BacktestResult(
        strategy=strategy,
        split=split,
        equity_curve=equity_curve,
        trades=trades,
        metrics=metrics,
        diagnostics={
            "null_signals_treated_as_flat": null_signals,
            "invalid_price_rows_skipped": skipped_prices,
        },
    )

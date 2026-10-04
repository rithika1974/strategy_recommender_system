"""Deterministic performance metrics for historical backtests."""

from __future__ import annotations

import math

import pandas as pd


def calculate_max_drawdown(equity: pd.Series) -> float:
    """Return maximum peak-to-trough loss as a positive fraction."""
    if equity.empty:
        return 0.0
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1
    return float(abs(drawdown.min()))


def calculate_sharpe(
    returns: pd.Series,
    *,
    annualization_days: int = 252,
    risk_free_rate: float = 0.0,
) -> float:
    """Return annualized Sharpe from chronological daily returns."""
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty or annualization_days <= 0:
        return 0.0
    daily_risk_free = (1 + risk_free_rate) ** (1 / annualization_days) - 1
    excess = clean - daily_risk_free
    standard_deviation = excess.std(ddof=1)
    if pd.isna(standard_deviation) or standard_deviation == 0:
        return 0.0
    return float(excess.mean() / standard_deviation * math.sqrt(annualization_days))


def compute_metrics(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    *,
    annualization_days: int = 252,
    risk_free_rate: float = 0.0,
) -> dict[str, float | int]:
    """Calculate portfolio-level return, risk, drawdown, and trade metrics."""
    if equity_curve.empty:
        return {
            "cumulative_return": 0.0,
            "cagr": 0.0,
            "volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "number_of_trades": 0,
        }
    curve = equity_curve.sort_values("date").reset_index(drop=True)
    equity = pd.to_numeric(curve["equity"], errors="raise")
    initial_equity = float(equity_curve.attrs.get("initial_cash", equity.iloc[0]))
    ending_equity = float(equity.iloc[-1])
    cumulative_return = ending_equity / initial_equity - 1
    dates = pd.to_datetime(curve["date"], errors="raise")
    elapsed_days = int((dates.iloc[-1] - dates.iloc[0]).days)
    years = elapsed_days / 365.25
    cagr = (
        (ending_equity / initial_equity) ** (1 / years) - 1
        if years > 0 and ending_equity > 0
        else 0.0
    )
    returns = equity.pct_change().fillna(0.0)
    volatility = (
        float(returns.std(ddof=1) * math.sqrt(annualization_days))
        if len(returns) > 1
        else 0.0
    )
    trade_frame = trades if trades is not None else pd.DataFrame()
    trade_count = len(trade_frame)
    win_rate = (
        float(pd.to_numeric(trade_frame["net_pnl"]).gt(0).mean())
        if trade_count and "net_pnl" in trade_frame
        else 0.0
    )
    return {
        "cumulative_return": float(cumulative_return),
        "cagr": float(cagr),
        "volatility": volatility,
        "sharpe": calculate_sharpe(
            returns,
            annualization_days=annualization_days,
            risk_free_rate=risk_free_rate,
        ),
        "max_drawdown": calculate_max_drawdown(equity),
        "win_rate": win_rate,
        "number_of_trades": trade_count,
    }

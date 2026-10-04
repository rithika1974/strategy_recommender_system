from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.backtesting.engine import BacktestConfig, SplitBoundary, run_backtest
from src.backtesting.execution import simulate_execution
from src.evaluation.metrics import calculate_max_drawdown, calculate_sharpe


def config(*, costs: float = 0.0, slippage: float = 0.0) -> BacktestConfig:
    return BacktestConfig(
        initial_cash=1_000.0,
        position_size=1.0,
        transaction_cost_bps=costs,
        slippage_bps=slippage,
        max_positions=1,
        annualization_days=252,
        risk_free_rate=0.0,
        splits={
            "train": SplitBoundary(date(2020, 1, 1), date(2020, 12, 31)),
            "validation": SplitBoundary(date(2021, 1, 1), date(2021, 12, 31)),
            "test": SplitBoundary(date(2022, 1, 1), None),
        },
    )


def prices(values: list[tuple[float, float]], start: str = "2020-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=len(values), freq="D")
    return pd.DataFrame(
        {
            "symbol": "TEST",
            "timestamp": dates,
            "open": [value[0] for value in values],
            "close": [value[1] for value in values],
        }
    )


def signals(values: list[str | None], start: str = "2020-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=len(values), freq="D")
    return pd.DataFrame(
        {
            "symbol": "TEST",
            "date": dates,
            "strategy": "test_strategy",
            "signal": values,
        }
    )


def test_next_open_long_entry_exit_and_portfolio_accounting():
    result = run_backtest(
        signals(["long", "flat", "flat"]),
        prices([(90, 90), (100, 105), (110, 110)]),
        strategy="test_strategy",
        split="train",
        config=config(),
    )

    trade = result.trades.iloc[0]
    assert trade["entry_date"] == "2020-01-02"
    assert trade["entry_price"] == 100
    assert trade["exit_date"] == "2020-01-03"
    assert trade["exit_price"] == 110
    assert trade["direction"] == "long"
    assert trade["gross_pnl"] == pytest.approx(100)
    assert trade["net_pnl"] == pytest.approx(100)
    assert result.equity_curve.iloc[-1]["equity"] == pytest.approx(1_100)


def test_short_entry_exit_and_pnl():
    result = run_backtest(
        signals(["short", "flat", "flat"]),
        prices([(90, 90), (100, 95), (80, 80)]),
        strategy="test_strategy",
        split="train",
        config=config(),
    )
    trade = result.trades.iloc[0]
    assert trade["entry_date"] == "2020-01-02"
    assert trade["exit_date"] == "2020-01-03"
    assert trade["direction"] == "short"
    assert trade["gross_pnl"] == pytest.approx(200)
    assert result.equity_curve.iloc[-1]["equity"] == pytest.approx(1_200)


def test_long_short_and_short_long_transitions_close_then_reopen():
    result = run_backtest(
        signals(["long", "short", "long", "flat", "flat"]),
        prices([(100, 100), (100, 100), (90, 90), (80, 80), (85, 85)]),
        strategy="test_strategy",
        split="train",
        config=config(),
    )
    assert list(result.trades["direction"]) == ["long", "short", "long"]
    assert list(result.trades["entry_date"]) == [
        "2020-01-02",
        "2020-01-03",
        "2020-01-04",
    ]
    assert list(result.trades["exit_date"]) == [
        "2020-01-03",
        "2020-01-04",
        "2020-01-05",
    ]


def test_transaction_costs_and_adverse_slippage():
    buy = simulate_execution(
        100, 10, "buy", slippage_bps=10, transaction_cost_bps=15
    )
    sell = simulate_execution(
        100, 10, "sell", slippage_bps=10, transaction_cost_bps=15
    )
    assert buy.execution_price == pytest.approx(100.1)
    assert sell.execution_price == pytest.approx(99.9)
    assert buy.transaction_cost == pytest.approx(100.1 * 10 * 0.0015)
    assert sell.transaction_cost == pytest.approx(99.9 * 10 * 0.0015)


def test_costs_flow_into_trade_net_pnl_and_final_equity():
    result = run_backtest(
        signals(["long", "flat", "flat"]),
        prices([(100, 100), (100, 100), (100, 100)]),
        strategy="test_strategy",
        split="train",
        config=config(costs=100),
    )
    trade = result.trades.iloc[0]
    assert trade["gross_pnl"] == pytest.approx(0)
    assert trade["net_pnl"] == pytest.approx(-trade["costs"])
    assert result.equity_curve.iloc[-1]["equity"] == pytest.approx(
        1_000 + trade["net_pnl"]
    )


def test_signal_never_executes_on_same_day_and_future_signal_cannot_change_past():
    market = prices([(50, 50), (100, 100), (200, 200)])
    original = run_backtest(
        signals(["long", "flat", "flat"]),
        market,
        strategy="test_strategy",
        split="train",
        config=config(),
    )
    changed = run_backtest(
        signals(["long", "flat", "short"]),
        market,
        strategy="test_strategy",
        split="train",
        config=config(),
    )
    assert original.trades.iloc[0]["entry_price"] == 100
    assert original.equity_curve.iloc[:2].equals(changed.equity_curve.iloc[:2])


def test_upstox_utc_timestamp_is_aligned_to_indian_trading_date():
    market = pd.DataFrame(
        {
            "symbol": ["TEST", "TEST", "TEST"],
            "timestamp": [
                "2019-12-31T18:30:00+00:00",
                "2020-01-01T18:30:00+00:00",
                "2020-01-02T18:30:00+00:00",
            ],
            "open": [50, 100, 110],
            "close": [50, 100, 110],
        }
    )
    result = run_backtest(
        signals(["long", "flat", "flat"]),
        market,
        strategy="test_strategy",
        split="train",
        config=config(),
    )
    assert result.trades.iloc[0]["entry_date"] == "2020-01-02"
    assert result.trades.iloc[0]["entry_price"] == 100


def test_equity_curve_drawdown_and_sharpe_metrics():
    assert calculate_max_drawdown(pd.Series([100.0, 120.0, 90.0, 110.0])) == pytest.approx(0.25)
    daily_returns = pd.Series([0.01, -0.01, 0.02])
    expected = daily_returns.mean() / daily_returns.std(ddof=1) * (252**0.5)
    assert calculate_sharpe(daily_returns) == pytest.approx(expected)


def test_empty_and_null_signals_are_flat_and_deterministic():
    market = prices([(100, 100), (101, 101), (102, 102)])
    empty = signals([])
    first = run_backtest(
        empty,
        market,
        strategy="test_strategy",
        split="train",
        config=config(),
    )
    second = run_backtest(
        empty,
        market,
        strategy="test_strategy",
        split="train",
        config=config(),
    )
    assert first.trades.empty
    assert first.equity_curve.equals(second.equity_curve)
    assert first.metrics == second.metrics

    null_result = run_backtest(
        signals([None, "flat", "flat"]),
        market,
        strategy="test_strategy",
        split="train",
        config=config(),
    )
    assert null_result.trades.empty
    assert null_result.diagnostics["null_signals_treated_as_flat"] == 1

    invalid_market = market.copy()
    invalid_market.loc[1, "open"] = None
    invalid_result = run_backtest(
        signals(["flat", "flat", "flat"]),
        invalid_market,
        strategy="test_strategy",
        split="train",
        config=config(),
    )
    assert invalid_result.diagnostics["invalid_price_rows_skipped"] == 1


def test_splits_are_isolated_and_open_position_is_closed_at_split_end():
    market = pd.concat(
        [prices([(100, 100), (100, 110)]), prices([(200, 200)], "2021-01-01")],
        ignore_index=True,
    )
    signal_frame = pd.concat(
        [signals(["long", "long"]), signals(["short"], "2021-01-01")],
        ignore_index=True,
    )
    result = run_backtest(
        signal_frame,
        market,
        strategy="test_strategy",
        split="train",
        config=config(),
    )
    assert set(result.equity_curve["date"]) == {"2020-01-01", "2020-01-02"}
    assert result.trades.iloc[-1]["exit_date"] == "2020-01-02"
    assert result.equity_curve.iloc[-1]["position"] == "flat"

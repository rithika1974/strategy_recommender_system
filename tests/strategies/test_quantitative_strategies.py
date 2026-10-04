from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from src.strategies.breakout import BreakoutStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.registry import (
    generate_signals_from_database,
    load_enabled_strategies,
)
from src.strategies.trend_following import TrendFollowingStrategy


def controlled_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["TEST"] * 4,
            "date": pd.date_range("2024-01-01", periods=4).strftime("%Y-%m-%d"),
            "return_20d": [0.10, -0.10, 0.01, None],
            "price_vs_sma20": [0.10, -0.10, 0.10, None],
            "sma20_vs_sma50": [0.10, -0.10, -0.10, 0.10],
            "market_trend_regime": ["BULL", "BEAR", "BULL", "BULL"],
            "rsi_14": [20.0, 80.0, 50.0, None],
            "bollinger_percent_b": [-0.1, 1.1, 0.5, 0.5],
            "breakout_flag": [1.0, 0.0, 0.0, None],
            "distance_from_low_20": [0.2, -0.1, 0.2, -0.1],
            "volume_ratio_20": [2.0, 2.0, 2.0, 2.0],
        }
    )


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        (MomentumStrategy({"lookback_days": 20, "threshold": 0.05}), ["long", "short", "flat", "flat"]),
        (
            TrendFollowingStrategy(
                {
                    "fast_window": 20,
                    "slow_window": 50,
                    "crossover_threshold": 0.0,
                    "require_regime_confirmation": True,
                }
            ),
            ["long", "short", "flat", "flat"],
        ),
        (
            MeanReversionStrategy(
                {
                    "rsi_window": 14,
                    "oversold_rsi": 30,
                    "overbought_rsi": 70,
                    "lower_percent_b": 0.0,
                    "upper_percent_b": 1.0,
                }
            ),
            ["long", "short", "flat", "flat"],
        ),
        (
            BreakoutStrategy({"lookback_days": 20, "breakout_multiplier": 1.5}),
            ["long", "short", "flat", "flat"],
        ),
    ],
)
def test_strategy_signal_contract_and_specific_conditions(strategy, expected):
    result = strategy.generate_signals(controlled_features())
    assert list(result.columns) == ["symbol", "date", "strategy", "signal"]
    assert result["signal"].tolist() == expected


@pytest.mark.parametrize(
    "strategy",
    [
        MomentumStrategy({"lookback_days": 20, "threshold": 0.05}),
        TrendFollowingStrategy(
            {
                "fast_window": 20,
                "slow_window": 50,
                "crossover_threshold": 0.0,
                "require_regime_confirmation": True,
            }
        ),
        MeanReversionStrategy(
            {
                "rsi_window": 14,
                "oversold_rsi": 30,
                "overbought_rsi": 70,
                "lower_percent_b": 0.0,
                "upper_percent_b": 1.0,
            }
        ),
        BreakoutStrategy({"lookback_days": 20, "breakout_multiplier": 1.5}),
    ],
)
def test_strategy_is_deterministic_and_does_not_read_future_rows(strategy):
    features = controlled_features()
    first = strategy.generate_signals(features)
    assert first.equals(strategy.generate_signals(features))

    changed_future = features.copy()
    for column in strategy.required_feature_columns():
        changed_future.loc[3, column] = 999 if column != "market_trend_regime" else "BEAR"
    second = strategy.generate_signals(changed_future)
    assert first.iloc[:3].equals(second.iloc[:3])


def test_configuration_parameters_change_signal_behavior():
    features = controlled_features()
    assert MomentumStrategy({"threshold": 0.05}).generate_signals(features).loc[0, "signal"] == "long"
    assert MomentumStrategy({"threshold": 0.20}).generate_signals(features).loc[0, "signal"] == "flat"
    assert BreakoutStrategy({"breakout_multiplier": 1.5}).generate_signals(features).loc[0, "signal"] == "long"
    assert BreakoutStrategy({"breakout_multiplier": 2.5}).generate_signals(features).loc[0, "signal"] == "flat"


def test_repository_configuration_loads_four_enabled_strategies():
    strategies = load_enabled_strategies()
    assert [strategy.name for strategy in strategies] == [
        "momentum",
        "trend_following",
        "mean_reversion",
        "breakout",
    ]


def test_database_signal_generation_does_not_modify_source_tables(tmp_path):
    database = tmp_path / "market_data.db"
    with sqlite3.connect(database) as connection:
        controlled_features().to_sql("features", connection, index=False)
        connection.execute(
            "CREATE TABLE augmentation_windows (sample_id TEXT PRIMARY KEY, symbol TEXT)"
        )
        connection.execute("INSERT INTO augmentation_windows VALUES ('sample-1', 'TEST')")
        before_features = connection.execute("SELECT * FROM features").fetchall()
        before_windows = connection.execute("SELECT * FROM augmentation_windows").fetchall()

    signals, diagnostics = generate_signals_from_database(database)

    assert len(signals) == 16
    assert diagnostics["momentum"]["rows_with_required_null"] == 1
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT * FROM features").fetchall() == before_features
        assert connection.execute("SELECT * FROM augmentation_windows").fetchall() == before_windows

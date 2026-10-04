from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import src.evaluation.optimization as optimization
from src.evaluation.optimization import OptimizationConfig, optimize_strategies
from src.evaluation.sensitivity import run_sensitivity_analysis
from src.evaluation.validation import calculate_objective, rank_candidates
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum import MomentumStrategy


def optimization_config(grids: dict) -> OptimizationConfig:
    return OptimizationConfig(
        sharpe_weight=1.0,
        return_weight=0.5,
        drawdown_weight=1.0,
        top_train_candidates=2,
        minimum_trade_count_warning=2,
        output_path=Path("unused.json"),
        parameter_grids=grids,
    )


def feature_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["TEST", "TEST"],
            "date": ["2020-01-01", "2020-01-02"],
            "return_20d": [0.04, 0.06],
            "rsi_14": [20.0, 80.0],
            "bollinger_percent_b": [-0.1, 1.1],
        }
    )


def fake_backtest_calls(monkeypatch):
    calls = []

    def fake_run_backtest(signals, prices, *, strategy, split, config):
        calls.append((strategy, split, int(signals["signal"].eq("long").sum())))
        long_count = int(signals["signal"].eq("long").sum())
        split_adjustment = 0.1 if split == "validation" else 0.0
        metrics = {
            "cumulative_return": long_count / 10 + split_adjustment,
            "cagr": 0.0,
            "volatility": 0.1,
            "sharpe": float(long_count),
            "max_drawdown": 0.1,
            "win_rate": 0.5,
            "number_of_trades": 10,
        }
        return SimpleNamespace(metrics=metrics)

    monkeypatch.setattr(optimization, "run_backtest", fake_run_backtest)
    return calls


def test_parameter_grid_generation_is_deterministic_and_includes_baseline():
    baseline = {"lookback_days": 20, "threshold": 0.05}
    grid = {"threshold": [0.03, 0.05, 0.08]}
    first = run_sensitivity_analysis(baseline, grid)
    second = run_sensitivity_analysis(baseline, grid)
    assert first == second
    assert len(first) == 3
    assert baseline in first


def test_repository_grids_have_expected_small_candidate_counts():
    config = optimization.load_optimization_config()
    baselines = {strategy.name: strategy.params for strategy in optimization.load_enabled_strategies()}
    counts = {
        name: len(run_sensitivity_analysis(baselines[name], grid))
        for name, grid in config.parameter_grids.items()
    }
    assert counts == {
        "momentum": 3,
        "trend_following": 6,
        "mean_reversion": 16,
        "breakout": 3,
    }


def test_risk_adjusted_objective_and_ranking_are_deterministic():
    metrics = {"sharpe": 1.0, "cumulative_return": 0.4, "max_drawdown": 0.2}
    assert calculate_objective(
        metrics, sharpe_weight=1, return_weight=0.5, drawdown_weight=1
    ) == pytest.approx(1.0)
    records = [
        {
            "objective": 1.0,
            "sharpe": 1.0,
            "cumulative_return": 0.2,
            "max_drawdown": 0.1,
            "number_of_trades": 10,
            "params": {"threshold": value},
        }
        for value in (0.08, 0.03)
    ]
    assert rank_candidates(records) == rank_candidates(list(reversed(records)))
    assert rank_candidates(records)[0]["params"] == {"threshold": 0.03}


def test_optimizer_uses_only_train_and_validation_and_passes_parameters(monkeypatch):
    calls = fake_backtest_calls(monkeypatch)
    baseline = MomentumStrategy({"lookback_days": 20, "threshold": 0.05})
    report = optimize_strategies(
        feature_rows(),
        pd.DataFrame(),
        backtest_config=None,
        optimization_config=optimization_config(
            {"momentum": {"threshold": [0.03, 0.05, 0.08]}}
        ),
        strategies=[baseline],
    )
    assert {split for _, split, _ in calls} == {"train", "validation"}
    assert "test" not in {record["split"] for record in report["candidate_results"]}
    assert {long_count for _, split, long_count in calls if split == "train"} == {0, 1, 2}
    assert report["methodology"]["excluded_split"] == "test"
    assert report["selections"]["momentum"]["baseline_params"] == baseline.params


def test_repeated_optimization_is_identical(monkeypatch):
    fake_backtest_calls(monkeypatch)
    arguments = {
        "features": feature_rows(),
        "prices": pd.DataFrame(),
        "backtest_config": None,
        "optimization_config": optimization_config(
            {"momentum": {"threshold": [0.03, 0.05, 0.08]}}
        ),
        "strategies": [MomentumStrategy({"lookback_days": 20, "threshold": 0.05})],
    }
    assert optimize_strategies(**arguments) == optimize_strategies(**arguments)


def test_invalid_strategy_parameter_combination_is_rejected(monkeypatch):
    fake_backtest_calls(monkeypatch)
    baseline = MeanReversionStrategy(
        {
            "rsi_window": 14,
            "oversold_rsi": 30,
            "overbought_rsi": 70,
            "lower_percent_b": 0.0,
            "upper_percent_b": 1.0,
        }
    )
    with pytest.raises(ValueError, match="RSI parameters"):
        optimize_strategies(
            feature_rows(),
            pd.DataFrame(),
            backtest_config=None,
            optimization_config=optimization_config(
                {
                    "mean_reversion": {
                        "oversold_rsi": [80],
                        "overbought_rsi": [70],
                    }
                }
            ),
            strategies=[baseline],
        )


def test_optimization_does_not_modify_database_source_tables(monkeypatch, tmp_path):
    fake_backtest_calls(monkeypatch)
    database = tmp_path / "market_data.db"
    with sqlite3.connect(database) as connection:
        feature_rows().to_sql("features", connection, index=False)
        connection.execute("CREATE TABLE augmentation_windows (sample_id TEXT PRIMARY KEY)")
        connection.execute("INSERT INTO augmentation_windows VALUES ('sample-1')")
        before_features = connection.execute("SELECT * FROM features").fetchall()
        before_windows = connection.execute("SELECT * FROM augmentation_windows").fetchall()
    optimize_strategies(
        feature_rows(),
        pd.DataFrame(),
        backtest_config=None,
        optimization_config=optimization_config(
            {"momentum": {"threshold": [0.03, 0.05]}}
        ),
        strategies=[MomentumStrategy({"lookback_days": 20, "threshold": 0.05})],
    )
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT * FROM features").fetchall() == before_features
        assert connection.execute("SELECT * FROM augmentation_windows").fetchall() == before_windows

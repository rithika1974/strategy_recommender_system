from __future__ import annotations

import copy
import sqlite3
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import src.evaluation.robustness as robustness
from scripts.run_robustness import _load_pretest_data
from src.backtesting.engine import BacktestConfig, BacktestResult, SplitBoundary
from src.evaluation.robustness import (
    RegimeThresholds,
    SensitivityThresholds,
    aggregate_stock_results,
    analyze_result_by_regime,
    assign_previous_regime,
    classify_sensitivity,
    find_neighboring_configurations,
    load_selected_strategies,
)
from src.evaluation.walk_forward import (
    TemporalFold,
    run_temporal_stability,
    validate_temporal_folds,
)
from src.strategies.momentum import MomentumStrategy


def backtest_config() -> BacktestConfig:
    return BacktestConfig(
        initial_cash=1_000,
        position_size=1.0,
        transaction_cost_bps=0,
        slippage_bps=0,
        max_positions=1,
        annualization_days=252,
        risk_free_rate=0,
        splits={"unused": SplitBoundary(date(2020, 1, 1), date(2025, 12, 31))},
    )


def sensitivity_thresholds(**changes) -> SensitivityThresholds:
    values = {
        "minimum_neighbors": 1,
        "minimum_validation_trades": 5,
        "maximum_objective_difference": 0.5,
        "maximum_return_difference": 0.1,
        "material_return_reversal": 0.05,
        "boundary_minimum_grid_values": 3,
    }
    values.update(changes)
    return SensitivityThresholds(**values)


def result(objective=1.0, return_value=0.1, trades=20):
    return {
        "objective": objective,
        "cumulative_return": return_value,
        "number_of_trades": trades,
    }


def test_repository_temporal_boundaries_do_not_overlap_or_access_2026():
    config = robustness.load_robustness_config()
    validate_temporal_folds(config.temporal_folds, config.test_cutoff)
    assert [fold.evaluation_start_date.year for fold in config.temporal_folds] == [
        2023,
        2024,
        2025,
    ]
    assert all(fold.evaluation_end_date < date(2026, 1, 1) for fold in config.temporal_folds)

    overlap = (
        TemporalFold(
            "one", date(2020, 1, 1), date(2022, 12, 31), date(2023, 1, 1), date(2023, 12, 31)
        ),
        TemporalFold(
            "two", date(2020, 1, 1), date(2022, 12, 31), date(2023, 6, 1), date(2024, 1, 1)
        ),
    )
    with pytest.raises(ValueError, match="overlap"):
        validate_temporal_folds(overlap, date(2026, 1, 1))
    frozen_test = (
        TemporalFold(
            "test", date(2020, 1, 1), date(2025, 12, 31), date(2026, 1, 1), date(2026, 12, 31)
        ),
    )
    with pytest.raises(ValueError, match="TEST"):
        validate_temporal_folds(frozen_test, date(2026, 1, 1))


def test_temporal_fold_starts_fresh_and_forces_end_liquidation():
    features = pd.DataFrame(
        {
            "symbol": ["TEST"] * 3,
            "date": pd.date_range("2023-01-01", periods=3),
            "return_20d": [0.2, 0.2, 0.2],
        }
    )
    prices = pd.DataFrame(
        {
            "symbol": ["TEST"] * 3,
            "timestamp": pd.date_range("2023-01-01", periods=3, tz="Asia/Kolkata"),
            "open": [100, 100, 110],
            "close": [100, 105, 115],
        }
    )
    strategy = MomentumStrategy({"lookback_days": 20, "threshold": 0.05})
    fold = TemporalFold(
        "fold", date(2020, 1, 1), date(2022, 12, 31), date(2023, 1, 1), date(2023, 12, 31)
    )
    before_features = features.copy(deep=True)
    before_prices = prices.copy(deep=True)
    rows = run_temporal_stability(
        features,
        prices,
        {"momentum": strategy},
        (fold,),
        backtest_config=backtest_config(),
        test_cutoff=date(2026, 1, 1),
    )
    assert rows[0]["fresh_initial_cash"] == 1_000
    assert rows[0]["ending_position"] == "flat"
    assert rows[0]["metrics"]["number_of_trades"] == 1
    pd.testing.assert_frame_equal(features, before_features)
    pd.testing.assert_frame_equal(prices, before_prices)


def test_selected_parameter_loading_and_neighbor_detection_are_exact():
    report = {
        "selections": {
            "momentum": {
                "selected_params": {"lookback_days": 20, "threshold": 0.05}
            }
        }
    }
    loaded = load_selected_strategies(report)
    assert loaded["momentum"].params == report["selections"]["momentum"]["selected_params"]
    neighbors = find_neighboring_configurations(
        loaded["momentum"].params,
        {"lookback_days": 20, "threshold": 0.05},
        {"threshold": [0.03, 0.05, 0.08]},
    )
    assert {item["threshold"] for item in neighbors} == {0.03, 0.08}


def test_sensitivity_classification_is_predeclared_and_deterministic():
    grid = {"threshold": [0.03, 0.05, 0.08]}
    selected = {"threshold": 0.05}
    stable = classify_sensitivity(
        result(), [result(0.8, 0.08)], selected, grid, sensitivity_thresholds()
    )
    assert stable[0] == "STABLE"
    assert stable == classify_sensitivity(
        result(), [result(0.8, 0.08)], selected, grid, sensitivity_thresholds()
    )
    assert classify_sensitivity(
        result(), [result(-0.2, -0.1)], selected, grid, sensitivity_thresholds()
    )[0] == "SENSITIVE"
    assert classify_sensitivity(
        result(), [result(0.9, 0.09)], {"threshold": 0.03}, grid, sensitivity_thresholds()
    )[0] == "BOUNDARY"
    assert classify_sensitivity(
        result(), [], selected, grid, sensitivity_thresholds()
    )[0] == "INSUFFICIENT_EVIDENCE"


def test_missing_neighbor_runs_validation_only_and_does_not_mutate_report(monkeypatch):
    selected_params = {"lookback_days": 20, "threshold": 0.05}
    report = {
        "selections": {
            "momentum": {
                "selected_params": selected_params,
                "baseline_params": selected_params,
                "selected_validation": result(),
            }
        },
        "candidate_results": [],
    }
    original = copy.deepcopy(report)
    calls = []

    def fake_backtest(signals, prices, *, strategy, split, config):
        calls.append(split)
        return SimpleNamespace(
            metrics={
                "cumulative_return": 0.08,
                "cagr": 0.08,
                "volatility": 0.1,
                "sharpe": 0.9,
                "max_drawdown": 0.1,
                "win_rate": 0.5,
                "number_of_trades": 20,
            }
        )

    monkeypatch.setattr(robustness, "run_backtest", fake_backtest)
    features = pd.DataFrame(
        {"symbol": ["TEST"], "date": ["2025-01-01"], "return_20d": [0.2]}
    )
    opt_config = SimpleNamespace(
        parameter_grids={"momentum": {"threshold": [0.03, 0.05, 0.08]}},
        sharpe_weight=1.0,
        return_weight=0.5,
        drawdown_weight=1.0,
    )
    output = robustness.run_parameter_sensitivity(
        features,
        pd.DataFrame(),
        {"momentum": MomentumStrategy(selected_params)},
        report,
        backtest_config=backtest_config(),
        optimization_config=opt_config,
        thresholds=sensitivity_thresholds(),
    )
    assert calls == ["validation", "validation"]
    assert all(row["source"] == "diagnostic_validation_run" for row in output["momentum"]["neighbors"])
    assert report == original


def test_stock_aggregation_reports_breadth_and_concentration():
    rows = [
        {"symbol": "A", "metrics": {"cumulative_return": 0.3, "sharpe": 1.0}},
        {"symbol": "B", "metrics": {"cumulative_return": 0.1, "sharpe": 0.2}},
        {"symbol": "C", "metrics": {"cumulative_return": -0.2, "sharpe": -1.0}},
    ]
    first = aggregate_stock_results(rows, top_fraction=1 / 3)
    second = aggregate_stock_results(list(reversed(rows)), top_fraction=1 / 3)
    assert first == second
    assert first["number_of_stocks"] == 3
    assert first["median_stock_return"] == pytest.approx(0.1)
    assert first["positive_return_percentage"] == pytest.approx(200 / 3)
    assert first["best_stock"]["symbol"] == "A"
    assert first["worst_stock"]["symbol"] == "C"
    assert first["positive_return_concentration_top_fraction"] == pytest.approx(0.75)


def test_regime_assignment_uses_strictly_previous_day_and_handles_small_samples():
    calendar = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "market_trend_regime": ["BULL", "BEAR"],
            "volatility_regime": ["LOW", "HIGH"],
        }
    )
    assigned = assign_previous_regime(
        pd.Series(["2025-01-01", "2025-01-02", "2025-01-03"]), calendar
    )
    assert pd.isna(assigned.iloc[0]["market_trend_regime"])
    assert assigned.iloc[1]["market_trend_regime"] == "BULL"
    assert assigned.iloc[2]["market_trend_regime"] == "BEAR"

    curve = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "equity": [1_000, 1_010, 1_000],
            "return": [0.0, 0.01, -10 / 1010],
        }
    )
    trades = pd.DataFrame(
        [{"entry_date": "2025-01-02", "net_pnl": 10.0}]
    )
    analysis, warnings = analyze_result_by_regime(
        BacktestResult("test", "analysis", curve, trades, {}, {}),
        calendar,
        thresholds=RegimeThresholds(2, 2, ("BULL", "BEAR"), ("LOW", "HIGH")),
        annualization_days=252,
        risk_free_rate=0,
    )
    assert analysis["BULL_LOW"]["number_of_observations"] == 1
    assert analysis["BULL_LOW"]["number_of_trades"] == 1
    assert analysis["BULL_LOW"]["status"] == "INSUFFICIENT_EVIDENCE"
    assert analysis["BULL_LOW"]["return"] is None
    assert warnings


def test_read_only_loader_leaves_features_augmentation_and_database_unchanged(tmp_path):
    database = tmp_path / "market_data.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE features (symbol TEXT, date TEXT)")
        connection.execute("INSERT INTO features VALUES ('A', '2025-01-01')")
        connection.execute("CREATE TABLE augmentation_windows (sample_id TEXT)")
        connection.execute("INSERT INTO augmentation_windows VALUES ('window-1')")
        connection.execute(
            "CREATE TABLE daily_prices (symbol TEXT, timestamp TEXT, open REAL, close REAL)"
        )
        connection.execute(
            "INSERT INTO daily_prices VALUES ('A', '2025-01-01T00:00:00+00:00', 1, 1)"
        )
    before = database.read_bytes()
    features, prices = _load_pretest_data(database)
    after = database.read_bytes()
    assert before == after
    assert len(features) == len(prices) == 1
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT * FROM augmentation_windows").fetchall() == [("window-1",)]


def test_complete_analysis_rejects_2026_before_any_strategy_or_backtest_access():
    config = robustness.load_robustness_config()
    features = pd.DataFrame({"date": ["2026-01-01"]})
    prices = pd.DataFrame(
        {
            "timestamp": ["2025-12-31T18:30:00+00:00"],
            "symbol": ["A"],
        }
    )
    with pytest.raises(ValueError, match="2026 TEST"):
        robustness.run_robustness_analysis(
            features,
            prices,
            {},
            backtest_config=backtest_config(),
            optimization_config=SimpleNamespace(),
            robustness_config=config,
        )

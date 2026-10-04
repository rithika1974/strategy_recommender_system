"""Run deterministic TRAIN/VALIDATION strategy parameter optimization."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtesting.engine import load_backtest_config
from src.evaluation.optimization import load_optimization_config, optimize_strategies


DATABASE_PATH = PROJECT_ROOT / "data" / "market_data.db"


def main() -> None:
    """Optimize without reading TEST through the backtesting interface."""
    with sqlite3.connect(DATABASE_PATH) as connection:
        features = pd.read_sql_query(
            "SELECT * FROM features WHERE date < '2026-01-01' ORDER BY symbol, date",
            connection,
        )
        prices = pd.read_sql_query(
            "SELECT symbol, timestamp, open, close FROM daily_prices "
            "WHERE timestamp < '2025-12-31T18:30:00+00:00' "
            "ORDER BY symbol, timestamp",
            connection,
        )
    optimization_config = load_optimization_config()
    report = optimize_strategies(
        features,
        prices,
        backtest_config=load_backtest_config(),
        optimization_config=optimization_config,
    )
    optimization_config.output_path.parent.mkdir(parents=True, exist_ok=True)
    optimization_config.output_path.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    summary = {
        "methodology": report["methodology"],
        "candidate_result_rows": len(report["candidate_results"]),
        "output_path": str(optimization_config.output_path),
        "selections": report["selections"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

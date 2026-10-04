"""Run deterministic V1 strategy backtests without persisting outputs."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtesting.engine import load_backtest_config, run_backtest
from src.strategies.registry import load_enabled_strategies


DATABASE_PATH = PROJECT_ROOT / "data" / "market_data.db"


def main() -> None:
    """Backtest each enabled strategy independently on each frozen split."""
    with sqlite3.connect(DATABASE_PATH) as connection:
        features = pd.read_sql_query(
            "SELECT * FROM features ORDER BY symbol, date", connection
        )
        prices = pd.read_sql_query(
            "SELECT symbol, timestamp, open, close FROM daily_prices "
            "ORDER BY symbol, timestamp",
            connection,
        )
    config = load_backtest_config()
    report: dict[str, dict] = {}
    for strategy in load_enabled_strategies():
        signals = strategy.generate_signals(features)
        report[strategy.name] = {}
        for split in ("train", "validation", "test"):
            result = run_backtest(
                signals,
                prices,
                strategy=strategy.name,
                split=split,
                config=config,
            )
            report[strategy.name][split] = {
                **result.metrics,
                "equity_rows": len(result.equity_curve),
                "diagnostics": result.diagnostics,
            }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

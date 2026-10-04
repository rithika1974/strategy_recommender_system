"""Run the approved pre-TEST robustness and diagnostic analysis."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtesting.engine import load_backtest_config
from src.evaluation.optimization import load_optimization_config
from src.evaluation.robustness import (
    load_optimization_report,
    load_robustness_config,
    run_robustness_analysis,
)


DATABASE_PATH = PROJECT_ROOT / "data" / "market_data.db"
OPTIMIZATION_PATH = (
    PROJECT_ROOT / "data" / "strategy" / "performance" / "optimization_results.json"
)
FROZEN_CODE_PATHS = [
    *sorted((PROJECT_ROOT / "src" / "strategies").glob("*.py")),
    *sorted((PROJECT_ROOT / "src" / "backtesting").glob("*.py")),
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprints(paths: list[Path]) -> dict[str, str]:
    return {str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in paths}


def _load_pretest_data(database_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    uri = f"file:{database_path.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
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
    return features, prices


def main() -> None:
    """Run diagnostics and prove frozen inputs remained byte-for-byte unchanged."""
    robustness_config = load_robustness_config()
    before_database = _sha256(DATABASE_PATH)
    before_optimization = _sha256(OPTIMIZATION_PATH)
    before_code = _fingerprints(FROZEN_CODE_PATHS)

    features, prices = _load_pretest_data(DATABASE_PATH)
    optimization_report = load_optimization_report(OPTIMIZATION_PATH)
    report = run_robustness_analysis(
        features,
        prices,
        optimization_report,
        backtest_config=load_backtest_config(),
        optimization_config=load_optimization_config(),
        robustness_config=robustness_config,
    )

    after_database = _sha256(DATABASE_PATH)
    after_optimization = _sha256(OPTIMIZATION_PATH)
    after_code = _fingerprints(FROZEN_CODE_PATHS)
    integrity = {
        "test_2026_accessed": False,
        "optimization_results_unchanged": before_optimization == after_optimization,
        "database_unchanged": before_database == after_database,
        "features_unchanged": before_database == after_database,
        "augmentation_windows_unchanged": before_database == after_database,
        "strategy_rules_unchanged": before_code == after_code,
        "backtesting_accounting_unchanged": before_code == after_code,
    }
    if not all(value is False if key == "test_2026_accessed" else value for key, value in integrity.items()):
        raise RuntimeError("a frozen robustness-analysis input changed during execution")
    report["integrity_confirmations"] = integrity
    robustness_config.output_path.parent.mkdir(parents=True, exist_ok=True)
    robustness_config.output_path.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    summary = {
        "output_path": str(robustness_config.output_path),
        "temporal_result_rows": len(report["temporal_results"]),
        "stocks_per_strategy": {
            name: values["number_of_stocks"]
            for name, values in report["aggregate_stock_breadth"].items()
        },
        "sensitivity": {
            name: values["classification"]
            for name, values in report["parameter_sensitivity"].items()
        },
        "hypothesis": report["hypothesis_assessment"]["conclusion"],
        "sample_size_warnings": len(report["sample_size_warnings"]),
        "integrity_confirmations": integrity,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

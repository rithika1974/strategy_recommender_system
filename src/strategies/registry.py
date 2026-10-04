from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.strategies.base import BaseStrategy
from src.strategies.breakout import BreakoutStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.trend_following import TrendFollowingStrategy


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "strategies.yaml"
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data" / "market_data.db"
STRATEGY_TYPES: dict[str, type[BaseStrategy]] = {
    "momentum": MomentumStrategy,
    "trend_following": TrendFollowingStrategy,
    "mean_reversion": MeanReversionStrategy,
    "breakout": BreakoutStrategy,
}


def load_enabled_strategies(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> list[BaseStrategy]:
    """Instantiate enabled deterministic strategies from the existing YAML file."""
    with Path(config_path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    entries = raw.get("strategies")
    if not isinstance(entries, list):
        raise ValueError("strategies configuration must contain a list")
    strategies: list[BaseStrategy] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or "name" not in entry:
            raise ValueError("each strategy configuration requires a name")
        name = str(entry["name"])
        if name in seen:
            raise ValueError(f"duplicate strategy configuration: {name}")
        seen.add(name)
        if name not in STRATEGY_TYPES:
            raise ValueError(f"unknown strategy: {name}")
        if entry.get("enabled", False):
            params: dict[str, Any] = entry.get("params") or {}
            if not isinstance(params, dict):
                raise ValueError(f"{name}.params must be a mapping")
            strategies.append(STRATEGY_TYPES[name](params))
    return strategies


def generate_all_signals(
    features: pd.DataFrame,
    strategies: list[BaseStrategy],
) -> pd.DataFrame:
    """Generate compact signal rows for each strategy without persisting them."""
    if not strategies:
        return pd.DataFrame(columns=["symbol", "date", "strategy", "signal"])
    return pd.concat(
        [strategy.generate_signals(features) for strategy in strategies],
        ignore_index=True,
    )


def generate_signals_from_database(
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Read DB2 features and return in-memory signals plus NULL diagnostics."""
    with sqlite3.connect(database_path) as connection:
        if not connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='features'"
        ).fetchone():
            raise ValueError("database does not contain the features table")
        features = pd.read_sql_query(
            "SELECT * FROM features ORDER BY symbol, date", connection
        )
    strategies = load_enabled_strategies(config_path)
    signals = generate_all_signals(features, strategies)
    null_diagnostics: dict[str, dict[str, int]] = {}
    for strategy in strategies:
        required = sorted(strategy.required_feature_columns())
        null_diagnostics[strategy.name] = {
            "rows_with_required_null": int(features[required].isna().any(axis=1).sum())
        }
    return signals, null_diagnostics

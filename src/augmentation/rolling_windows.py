from __future__ import annotations

import hashlib
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "augmentation.yaml"
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data" / "market_data.db"
REQUIRED_COLUMNS = {
    "symbol",
    "date",
    "market_trend_regime",
    "volatility_regime",
}
SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class DateSplit:
    """Inclusive calendar boundaries for one chronological dataset split."""

    start_date: date
    end_date: date | None


@dataclass(frozen=True)
class AugmentationConfig:
    """Validated rolling-window configuration."""

    window_length: int
    training_stride: int
    evaluation_stride: int
    train: DateSplit
    validation: DateSplit
    test: DateSplit
    regime_source: str
    null_handling: str
    ignored_null_columns: tuple[str, ...]
    seed: int


def _parse_date(value: Any, field: str, *, optional: bool = False) -> date | None:
    if value is None and optional:
        return None
    if value is None:
        raise ValueError(f"{field} must be configured")
    try:
        return pd.Timestamp(value).date()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a valid date") from exc


def _parse_split(raw: dict[str, Any], name: str) -> DateSplit:
    if not isinstance(raw, dict):
        raise ValueError(f"augmentation.{name} must be a mapping")
    start = _parse_date(raw.get("start_date"), f"{name}.start_date")
    end = _parse_date(raw.get("end_date"), f"{name}.end_date", optional=True)
    if end is not None and end < start:
        raise ValueError(f"{name}.end_date cannot be before its start_date")
    return DateSplit(start_date=start, end_date=end)


def _validate_split_order(config: AugmentationConfig) -> None:
    if config.train.end_date is None or config.validation.end_date is None:
        raise ValueError("train and validation must have explicit end dates")
    if config.train.end_date >= config.validation.start_date:
        raise ValueError("train and validation date ranges overlap")
    if config.validation.end_date >= config.test.start_date:
        raise ValueError("validation and test date ranges overlap")


def load_augmentation_config(path: str | Path = DEFAULT_CONFIG_PATH) -> AugmentationConfig:
    """Load and validate augmentation settings using the existing YAML convention."""
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = (yaml.safe_load(handle) or {}).get("augmentation", {})
    try:
        config = AugmentationConfig(
            window_length=int(raw["window_length"]),
            training_stride=int(raw["training_stride"]),
            evaluation_stride=int(raw["evaluation_stride"]),
            train=_parse_split(raw.get("train"), "train"),
            validation=_parse_split(raw.get("validation"), "validation"),
            test=_parse_split(raw.get("test"), "test"),
            regime_source=str(raw["regime_source"]),
            null_handling=str(raw["null_handling"]),
            ignored_null_columns=tuple(raw.get("ignored_null_columns", ())),
            seed=int(raw["seed"]),
        )
    except KeyError as exc:
        raise ValueError(f"Missing augmentation setting: {exc.args[0]}") from exc
    if config.window_length <= 0 or config.training_stride <= 0 or config.evaluation_stride <= 0:
        raise ValueError("window length and strides must be positive")
    if config.regime_source != "window_end":
        raise ValueError("Only window_end regime assignment is supported")
    if config.null_handling != "reject_active_feature_nulls":
        raise ValueError("Only reject_active_feature_nulls is supported")
    _validate_split_order(config)
    return config


def initialize_augmentation_table(connection: sqlite3.Connection) -> None:
    """Create the compact window-reference table without changing features."""
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS augmentation_windows (
            sample_id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            split TEXT NOT NULL CHECK (split IN ('train', 'validation', 'test')),
            window_start_date TEXT NOT NULL,
            window_end_date TEXT NOT NULL,
            window_length INTEGER NOT NULL,
            stride INTEGER NOT NULL,
            market_trend_regime TEXT NOT NULL,
            volatility_regime TEXT NOT NULL,
            seed INTEGER NOT NULL,
            UNIQUE (
                symbol, split, window_start_date, window_end_date,
                window_length, stride, seed
            )
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_augmentation_windows_symbol_split "
        "ON augmentation_windows(symbol, split, window_start_date)"
    )


def _sample_id(
    symbol: str,
    split: str,
    start_date: str,
    end_date: str,
    window_length: int,
    stride: int,
    seed: int,
) -> str:
    value = "|".join(
        map(str, (symbol, split, start_date, end_date, window_length, stride, seed))
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _split_frame(frame: pd.DataFrame, split: DateSplit) -> pd.DataFrame:
    mask = frame["_parsed_date"].dt.date >= split.start_date
    if split.end_date is not None:
        mask &= frame["_parsed_date"].dt.date <= split.end_date
    return frame.loc[mask].reset_index(drop=True)


def generate_window_metadata(
    features: pd.DataFrame,
    config: AugmentationConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate deterministic metadata for valid chronological feature windows."""
    missing = REQUIRED_COLUMNS.difference(features.columns)
    if missing:
        raise ValueError(f"features table is missing required columns: {sorted(missing)}")
    unknown_ignored = set(config.ignored_null_columns).difference(features.columns)
    if unknown_ignored:
        raise ValueError(f"ignored NULL columns are not in features: {sorted(unknown_ignored)}")

    frame = features.copy()
    frame["_parsed_date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame.duplicated(["symbol", "date"]).any():
        raise ValueError("features contains duplicate (symbol, date) rows")
    active_columns = [
        column
        for column in features.columns
        if column not in set(config.ignored_null_columns)
    ]

    windows: list[dict[str, Any]] = []
    rejection_reasons: Counter[str] = Counter()
    rejection_columns: Counter[str] = Counter()
    candidate_count = 0

    split_configs = {
        "train": config.train,
        "validation": config.validation,
        "test": config.test,
    }
    for symbol, symbol_frame in frame.groupby("symbol", sort=True):
        symbol_frame = symbol_frame.sort_values("_parsed_date").reset_index(drop=True)
        if not symbol_frame["_parsed_date"].is_monotonic_increasing:
            raise ValueError(f"dates are not chronological for {symbol}")
        for split_name in SPLIT_NAMES:
            split_frame = _split_frame(symbol_frame, split_configs[split_name])
            stride = (
                config.training_stride
                if split_name == "train"
                else config.evaluation_stride
            )
            last_start = len(split_frame) - config.window_length
            for start in range(0, last_start + 1, stride):
                candidate_count += 1
                window = split_frame.iloc[start : start + config.window_length]
                null_columns = [column for column in active_columns if window[column].isna().any()]
                if null_columns:
                    rejection_reasons["active_feature_null"] += 1
                    rejection_columns.update(null_columns)
                    continue
                start_date = str(window.iloc[0]["date"])
                end_date = str(window.iloc[-1]["date"])
                end_row = window.iloc[-1]
                windows.append(
                    {
                        "sample_id": _sample_id(
                            str(symbol),
                            split_name,
                            start_date,
                            end_date,
                            config.window_length,
                            stride,
                            config.seed,
                        ),
                        "symbol": str(symbol),
                        "split": split_name,
                        "window_start_date": start_date,
                        "window_end_date": end_date,
                        "window_length": config.window_length,
                        "stride": stride,
                        "market_trend_regime": str(end_row["market_trend_regime"]),
                        "volatility_regime": str(end_row["volatility_regime"]),
                        "seed": config.seed,
                    }
                )

    if len({window["sample_id"] for window in windows}) != len(windows):
        raise ValueError("generated duplicate sample IDs")
    summary = {
        "total_feature_rows": len(features),
        "symbols_processed": int(features["symbol"].nunique()),
        "candidate_windows": candidate_count,
        "rejected_windows": sum(rejection_reasons.values()),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "rejection_columns": dict(sorted(rejection_columns.items())),
    }
    return windows, summary


def _summarize_windows(windows: list[dict[str, Any]]) -> dict[str, Any]:
    split_counts = Counter(window["split"] for window in windows)
    symbol_counts = Counter(window["symbol"] for window in windows)
    regime_counts = Counter(
        f"{window['market_trend_regime']}|{window['volatility_regime']}"
        for window in windows
    )
    return {
        "total_windows": len(windows),
        "windows_per_split": {name: split_counts[name] for name in SPLIT_NAMES},
        "windows_per_symbol": dict(sorted(symbol_counts.items())),
        "regime_distribution": dict(sorted(regime_counts.items())),
        "earliest_window_start": min(
            (window["window_start_date"] for window in windows), default=None
        ),
        "latest_window_end": max(
            (window["window_end_date"] for window in windows), default=None
        ),
    }


def build_augmentation_windows(
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Rebuild compact augmentation references from the existing features table."""
    config = load_augmentation_config(config_path)
    with sqlite3.connect(database_path) as connection:
        if not connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='features'"
        ).fetchone():
            raise ValueError("database does not contain the features table")
        features = pd.read_sql_query(
            "SELECT * FROM features ORDER BY symbol, date", connection
        )
        windows, summary = generate_window_metadata(features, config)
        initialize_augmentation_table(connection)
        connection.execute("DELETE FROM augmentation_windows")
        connection.executemany(
            """
            INSERT INTO augmentation_windows (
                sample_id, symbol, split, window_start_date, window_end_date,
                window_length, stride, market_trend_regime,
                volatility_regime, seed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    window["sample_id"],
                    window["symbol"],
                    window["split"],
                    window["window_start_date"],
                    window["window_end_date"],
                    window["window_length"],
                    window["stride"],
                    window["market_trend_regime"],
                    window["volatility_regime"],
                    window["seed"],
                )
                for window in windows
            ],
        )
        connection.commit()
    return {
        **summary,
        **_summarize_windows(windows),
        "window_length": config.window_length,
        "training_stride": config.training_stride,
        "evaluation_stride": config.evaluation_stride,
        "splits": {
            name: {
                "start_date": getattr(config, name).start_date.isoformat(),
                "end_date": (
                    getattr(config, name).end_date.isoformat()
                    if getattr(config, name).end_date is not None
                    else None
                ),
            }
            for name in SPLIT_NAMES
        },
        "seed": config.seed,
    }

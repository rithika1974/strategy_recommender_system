from __future__ import annotations

import logging
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

from src.data.instrument_resolver import InstrumentResolutionError, UpstoxInstrumentResolver
from src.data.normalizer import normalize_raw_response
from src.data.providers.upstox import (
    UpstoxAuthenticationError,
    UpstoxProvider,
    UpstoxProviderError,
)
from src.data.validators import DataValidationError, validate_daily_frame

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UNIVERSE_PATH = PROJECT_ROOT / "configs" / "universe.yaml"
DATA_SOURCES_PATH = PROJECT_ROOT / "configs" / "data_sources.yaml"
DATABASE_PATH = PROJECT_ROOT / "data" / "market_data.db"
INSTRUMENT_MAPPING_PATH = PROJECT_ROOT / "data" / "metadata" / "instruments" / "upstox_equity_mapping.json"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _normalize_iso_date(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, str):
        return pd.Timestamp(value).date().isoformat()
    raise TypeError(f"Unsupported date value type: {type(value)!r}")


def initialize_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS stocks (
            symbol TEXT PRIMARY KEY,
            company TEXT NOT NULL,
            category TEXT NOT NULL,
            instrument_key TEXT,
            isin TEXT
        );

        CREATE TABLE IF NOT EXISTS daily_prices (
            symbol TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            open_interest REAL NOT NULL,
            PRIMARY KEY (symbol, timestamp),
            FOREIGN KEY (symbol) REFERENCES stocks(symbol)
        );
        """
    )


def _insert_stock(connection: sqlite3.Connection, stock: dict[str, Any]) -> None:
    connection.execute(
        """
        INSERT INTO stocks (symbol, company, category, instrument_key, isin)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(symbol) DO UPDATE SET
            company = excluded.company,
            category = excluded.category,
            instrument_key = excluded.instrument_key,
            isin = excluded.isin
        """,
        (
            stock["symbol"],
            stock["company"],
            stock["category"],
            stock.get("instrument_key"),
            stock.get("isin"),
        ),
    )


def _insert_prices(connection: sqlite3.Connection, symbol: str, frame: pd.DataFrame) -> int:
    rows = [
        (
            symbol,
            timestamp.isoformat(),
            float(row["open"]),
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
            float(row["volume"]),
            float(row["open_interest"]),
        )
        for timestamp, row in frame.set_index("timestamp").iterrows()
    ]
    changes_before = connection.total_changes
    connection.executemany(
        """
        INSERT OR IGNORE INTO daily_prices
        (symbol, timestamp, open, high, low, close, volume, open_interest)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return connection.total_changes - changes_before


def _latest_stored_date(connection: sqlite3.Connection, symbol: str) -> date | None:
    row = connection.execute(
        "SELECT MAX(timestamp) FROM daily_prices WHERE symbol = ?", (symbol,)
    ).fetchone()
    if not row or row[0] is None:
        return None
    return pd.Timestamp(row[0]).date()


def _download_stock_data(
    stock: dict[str, Any],
    source_config: dict[str, Any],
    resolver: UpstoxInstrumentResolver,
    client: UpstoxProvider,
    connection: sqlite3.Connection,
) -> tuple[int, str]:
    resolved_stock = resolver.resolve(stock)
    symbol = resolved_stock["symbol"]
    _insert_stock(connection, resolved_stock)
    configured_from_date = _normalize_iso_date(
        resolved_stock.get("from_date")
        or source_config.get("start_date")
        or _load_yaml(UNIVERSE_PATH).get("universe", {}).get("start_date")
    )
    configured_to_date = _normalize_iso_date(
        resolved_stock.get("to_date")
        or source_config.get("end_date")
        or _load_yaml(UNIVERSE_PATH).get("universe", {}).get("end_date")
        or date.today()
    )
    latest_stored_date = _latest_stored_date(connection, symbol)
    from_date = (
        (latest_stored_date + timedelta(days=1)).isoformat()
        if latest_stored_date is not None
        else configured_from_date
    )
    to_date = configured_to_date
    if not from_date or not to_date:
        raise ValueError("Both from_date and to_date are required")
    if latest_stored_date is not None and from_date > to_date:
        return 0, "up_to_date"

    payload = client.get_daily_history(resolved_stock["instrument_key"], from_date, to_date)
    data = payload.get("data") if isinstance(payload, dict) else None
    candles = (
        data.get("candles")
        if isinstance(data, dict)
        else payload.get("candles")
        if isinstance(payload, dict)
        else None
    )
    if latest_stored_date is not None and candles == []:
        return 0, "up_to_date"
    frame = normalize_raw_response(payload)
    validate_daily_frame(frame, from_date=from_date, to_date=to_date)
    inserted_count = _insert_prices(connection, symbol, frame)
    return inserted_count, "updated" if latest_stored_date is not None else "downloaded"


def download_all_stocks(
    *,
    database_path: str | Path = DATABASE_PATH,
    refresh: bool = False,
    symbols: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Download the configured Dataset 1 universe, optionally scoped to symbols."""
    universe = _load_yaml(UNIVERSE_PATH)
    source_config = _load_yaml(DATA_SOURCES_PATH)
    stock_configs = universe.get("universe", {}).get("stocks", [])
    requested_symbols = set(symbols) if symbols is not None else None
    if requested_symbols is not None:
        stock_configs = [
            stock for stock in stock_configs if stock["symbol"] in requested_symbols
        ]
    database = Path(database_path)
    database.parent.mkdir(parents=True, exist_ok=True)
    resolver = UpstoxInstrumentResolver(cache_path=INSTRUMENT_MAPPING_PATH)
    client = UpstoxProvider()
    results: list[dict[str, Any]] = []

    with sqlite3.connect(database) as connection:
        initialize_database(connection)
        for stock in stock_configs:
            symbol = stock["symbol"]
            try:
                _insert_stock(connection, stock)
                connection.commit()
                if refresh:
                    connection.execute("DELETE FROM daily_prices WHERE symbol = ?", (symbol,))
                records, status = _download_stock_data(
                    stock, source_config, resolver, client, connection
                )
                connection.commit()
                results.append({"symbol": symbol, "records": records, "status": status})
            except UpstoxAuthenticationError:
                connection.rollback()
                raise
            except (
                DataValidationError,
                InstrumentResolutionError,
                UpstoxProviderError,
                ValueError,
                TypeError,
            ) as exc:
                connection.rollback()
                results.append({"symbol": symbol, "records": 0, "status": f"failed: {exc}"})
                logger.error("Failed %s: %s", symbol, exc)

    return {"total": len(stock_configs), "results": results, "database": str(database)}

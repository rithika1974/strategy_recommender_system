from __future__ import annotations

import gzip
import json
import sqlite3
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.normalizer import normalize_raw_response
from src.data.providers.upstox import UpstoxAuthenticationError, UpstoxProvider
from src.data.validators import validate_daily_frame

DATABASE_PATH = PROJECT_ROOT / "data" / "market_data.db"
MASTER_URL = "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz"
GLOBAL_MASTER_URL = "https://assets.upstox.com/market-quote/instruments/exchange/global.json.gz"
INDEXES = {"NIFTY 50": ("NIFTY", "Nifty 50"), "INDIA VIX": ("INDIA VIX", "India VIX")}
GLOBAL_INSTRUMENTS = {"USDINR": ("USD INR", "USDINR"), "BRENT": ("Oil (Brent)", "BZUSD")}


def initialize_raw_tables(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS market_data (
            source TEXT NOT NULL,
            symbol TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL NOT NULL,
            volume REAL,
            open_interest REAL,
            PRIMARY KEY (source, symbol, timestamp)
        );
        CREATE INDEX IF NOT EXISTS idx_market_data_symbol_date
            ON market_data(symbol, timestamp);
        CREATE TABLE IF NOT EXISTS macro_data (
            source TEXT NOT NULL,
            indicator TEXT NOT NULL,
            date TEXT NOT NULL,
            value REAL NOT NULL,
            PRIMARY KEY (source, indicator, date)
        );
        """
    )


def resolve_indexes() -> dict[str, str]:
    response = requests.get(MASTER_URL, timeout=60)
    response.raise_for_status()
    master = json.loads(gzip.decompress(response.content))
    resolved: dict[str, str] = {}
    for symbol, (trading_symbol, name) in INDEXES.items():
        matches = [
            item for item in master
            if item.get("segment") == "NSE_INDEX"
            and item.get("trading_symbol") == trading_symbol
            and item.get("name") == name
            and item.get("instrument_type") == "INDEX"
        ]
        if len(matches) != 1:
            raise ValueError(f"{symbol}: expected one official Upstox index mapping, found {len(matches)}")
        resolved[symbol] = matches[0]["instrument_key"]
    return resolved


def resolve_global_instruments() -> dict[str, str]:
    response = requests.get(GLOBAL_MASTER_URL, timeout=60)
    response.raise_for_status()
    master = json.loads(gzip.decompress(response.content))
    resolved: dict[str, str] = {}
    for symbol, (name, trading_symbol) in GLOBAL_INSTRUMENTS.items():
        matches = [
            item for item in master
            if item.get("segment") == "GLOBAL_INDICATOR"
            and item.get("name") == name
            and item.get("trading_symbol") == trading_symbol
        ]
        if len(matches) != 1:
            raise ValueError(f"{symbol}: expected one official Upstox global mapping, found {len(matches)}")
        resolved[symbol] = matches[0]["instrument_key"]
    return resolved


def ingest_indexes(connection: sqlite3.Connection) -> dict[str, int | str]:
    keys = {**resolve_indexes(), **resolve_global_instruments()}
    provider = UpstoxProvider()
    results: dict[str, int | str] = {}
    for symbol, instrument_key in keys.items():
        connection.execute("SAVEPOINT market_instrument")
        try:
            payload = provider.get_daily_history(instrument_key, "2020-01-01", date.today().isoformat())
            frame = normalize_raw_response(payload)
            invalid = (
                (frame["high"] < frame["open"])
                | (frame["high"] < frame["close"])
                | (frame["low"] > frame["open"])
                | (frame["low"] > frame["close"])
                | (frame["high"] < frame["low"])
            )
            if invalid.any():
                print(f"MARKET {symbol}: skipped {int(invalid.sum())} source rows with invalid OHLC")
                frame = frame.loc[~invalid].copy()
            validate_daily_frame(frame, from_date="2020-01-01", to_date=date.today().isoformat())
            rows = [
                (
                    "upstox",
                    symbol,
                    row.timestamp.isoformat(),
                    row.open,
                    row.high,
                    row.low,
                    row.close,
                    row.volume,
                    row.open_interest,
                )
                for row in frame.itertuples(index=False)
            ]
            connection.executemany(
                """
                INSERT OR IGNORE INTO market_data
                (source, symbol, timestamp, open, high, low, close, volume, open_interest)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            connection.execute("RELEASE SAVEPOINT market_instrument")
            results[symbol] = len(rows)
        except UpstoxAuthenticationError:
            connection.execute("ROLLBACK TO SAVEPOINT market_instrument")
            connection.execute("RELEASE SAVEPOINT market_instrument")
            raise
        except Exception as exc:
            connection.execute("ROLLBACK TO SAVEPOINT market_instrument")
            connection.execute("RELEASE SAVEPOINT market_instrument")
            results[symbol] = f"failed: {exc}"
    return results


def validate_raw_tables(connection: sqlite3.Connection) -> list[str]:
    errors: list[str] = []
    for table in ("market_data", "macro_data"):
        if not connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone():
            errors.append(f"missing table: {table}")
    duplicate_count = connection.execute(
        "SELECT COUNT(*) FROM (SELECT source, symbol, timestamp, COUNT(*) n "
        "FROM market_data GROUP BY source, symbol, timestamp HAVING n > 1)"
    ).fetchone()[0]
    if duplicate_count:
        errors.append(f"market_data duplicate groups: {duplicate_count}")
    invalid_ohlc = connection.execute(
        "SELECT COUNT(*) FROM market_data "
        "WHERE high < open OR high < close OR low > open OR low > close OR high < low"
    ).fetchone()[0]
    if invalid_ohlc:
        errors.append(f"market_data invalid OHLC rows: {invalid_ohlc}")
    return errors


def main() -> None:
    with sqlite3.connect(DATABASE_PATH) as connection:
        initialize_raw_tables(connection)
        try:
            results = ingest_indexes(connection)
            connection.commit()
            for symbol, count in results.items():
                print(f"MARKET {symbol}: {count} rows")
        except Exception as exc:
            connection.rollback()
            print(f"MARKET INGESTION FAILED: {exc}")
        errors = validate_raw_tables(connection)
        print("RAW VALIDATION: PASS" if not errors else "RAW VALIDATION: " + "; ".join(errors))


if __name__ == "__main__":
    main()

"""Replaceable data-source boundary for quantitative analysis."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data" / "market_data.db"


@runtime_checkable
class AnalysisDataSource(Protocol):
    """Data capabilities needed by the service, independent of vendor/source."""

    @property
    def source_name(self) -> str: ...

    @property
    def frequency(self) -> str: ...

    def list_stocks(self) -> list[dict]: ...

    def stock_exists(self, symbol: str) -> bool: ...

    def load_features(self, symbol: str) -> pd.DataFrame: ...

    def load_prices(self, symbol: str) -> pd.DataFrame: ...


class SQLiteAnalysisDataSource:
    """Read-only adapter over the existing unified historical database."""

    def __init__(self, database_path: str | Path = DEFAULT_DATABASE_PATH) -> None:
        self.database_path = Path(database_path)

    @property
    def source_name(self) -> str:
        return "upstox_historical_sqlite"

    @property
    def frequency(self) -> str:
        return "daily"

    def _connect(self) -> sqlite3.Connection:
        if not self.database_path.exists():
            raise FileNotFoundError(f"market database not found: {self.database_path}")
        return sqlite3.connect(
            f"file:{self.database_path.resolve().as_posix()}?mode=ro", uri=True
        )

    def list_stocks(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT symbol, company, category, instrument_key, isin "
                "FROM stocks ORDER BY symbol"
            ).fetchall()
        return [
            {
                "symbol": row[0],
                "company": row[1],
                "category": row[2],
                "instrument_key": row[3],
                "isin": row[4],
            }
            for row in rows
        ]

    def stock_exists(self, symbol: str) -> bool:
        with self._connect() as connection:
            return (
                connection.execute(
                    "SELECT 1 FROM stocks WHERE symbol = ?", (symbol,)
                ).fetchone()
                is not None
            )

    def load_features(self, symbol: str) -> pd.DataFrame:
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='features'"
            ).fetchone()
            if not exists:
                return pd.DataFrame()
            return pd.read_sql_query(
                "SELECT * FROM features WHERE symbol = ? ORDER BY date",
                connection,
                params=(symbol,),
            )

    def load_prices(self, symbol: str) -> pd.DataFrame:
        with self._connect() as connection:
            return pd.read_sql_query(
                "SELECT symbol, timestamp, open, high, low, close, volume, "
                "open_interest FROM daily_prices WHERE symbol = ? ORDER BY timestamp",
                connection,
                params=(symbol,),
            )

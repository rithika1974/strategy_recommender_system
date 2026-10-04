from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from collections.abc import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.market import calculate_market_features
from src.features.macro import calculate_macro_features
from src.features.regime import calculate_market_trend_regime, calculate_volatility_regime
from src.features.technical import (
    calculate_atr, calculate_bollinger_bands, calculate_bollinger_percent_b,
    calculate_breakout_features, calculate_ema, calculate_macd, calculate_price_vs_sma,
    calculate_returns, calculate_rsi, calculate_sma, calculate_sma_difference,
    calculate_volatility, calculate_volume_ratio,
)

DATABASE_PATH = PROJECT_ROOT / "data" / "market_data.db"
FEATURE_COLUMNS = [
    "symbol", "date", "return_1d", "return_5d", "return_20d", "sma_20", "sma_50", "ema_20",
    "price_vs_sma20", "price_vs_sma50", "sma20_vs_sma50", "rsi_14", "macd", "macd_signal",
    "atr_14", "volatility_20", "volume_ratio_20", "bollinger_upper", "bollinger_lower",
    "bollinger_percent_b", "high_20", "low_20", "distance_from_high_20", "distance_from_low_20",
    "breakout_flag", "nifty_return_1d", "nifty_return_20d", "relative_strength_20d", "india_vix",
    "usdinr_return_1d", "brent_return_1d", "repo_rate", "cpi", "market_trend_regime",
    "volatility_regime",
]


def _optional_table(connection: sqlite3.Connection, table: str) -> pd.DataFrame:
    if not connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone():
        return pd.DataFrame()
    return pd.read_sql_query(f"SELECT * FROM {table}", connection)


def _market_frame(market: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if market.empty:
        return pd.DataFrame()
    frame = market[market["symbol"] == symbol].copy()
    if frame.empty:
        return frame
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.sort_values("timestamp").reset_index(drop=True)


def _technical_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = calculate_returns(frame)
    result["sma_20"] = calculate_sma(frame, 20)
    result["sma_50"] = calculate_sma(frame, 50)
    result["ema_20"] = calculate_ema(frame, 20)
    result["price_vs_sma20"] = calculate_price_vs_sma(frame, 20)
    result["price_vs_sma50"] = calculate_price_vs_sma(frame, 50)
    result["sma20_vs_sma50"] = calculate_sma_difference(frame)
    result["rsi_14"] = calculate_rsi(frame)
    result[["macd", "macd_signal"]] = calculate_macd(frame)
    result["atr_14"] = calculate_atr(frame)
    result["volatility_20"] = calculate_volatility(frame)
    result["volume_ratio_20"] = calculate_volume_ratio(frame)
    result[["bollinger_upper", "bollinger_lower"]] = calculate_bollinger_bands(frame)
    result["bollinger_percent_b"] = calculate_bollinger_percent_b(frame)
    result[["high_20", "low_20", "distance_from_high_20", "distance_from_low_20", "breakout_flag"]] = calculate_breakout_features(frame)
    return result


def build_features(
    database_path: str | Path = DATABASE_PATH,
    symbols: Iterable[str] | None = None,
) -> int:
    """Build Dataset 2 features, rebuilding all or replacing selected symbols."""
    with sqlite3.connect(database_path) as connection:
        selected_symbols = list(dict.fromkeys(symbols)) if symbols is not None else None
        if selected_symbols:
            placeholders = ", ".join("?" for _ in selected_symbols)
            prices = pd.read_sql_query(
                f"SELECT * FROM daily_prices WHERE symbol IN ({placeholders}) "
                "ORDER BY symbol, timestamp",
                connection,
                params=selected_symbols,
            )
        else:
            prices = pd.read_sql_query(
                "SELECT * FROM daily_prices ORDER BY symbol, timestamp", connection
            )
        market = _optional_table(connection, "market_data")
        if selected_symbols:
            if connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='features'"
            ).fetchone():
                connection.executemany(
                    "DELETE FROM features WHERE symbol = ?",
                    [(symbol,) for symbol in selected_symbols],
                )
            else:
                selected_symbols = None
        if selected_symbols is None:
            connection.execute("DROP TABLE IF EXISTS features")
            types = ", ".join(
                f"{column} {'TEXT' if column in {'symbol', 'date', 'market_trend_regime', 'volatility_regime'} else 'REAL'}"
                for column in FEATURE_COLUMNS
            )
            connection.execute(f"CREATE TABLE features ({types}, PRIMARY KEY (symbol, date))")
        if prices.empty:
            connection.commit()
            return 0
        all_rows: list[dict] = []
        nifty = _market_frame(market, "NIFTY 50")
        vix = _market_frame(market, "INDIA VIX")
        usdinr = _market_frame(market, "USDINR")
        brent = _market_frame(market, "BRENT")
        for symbol, stock in prices.groupby("symbol", sort=True):
            stock = stock.copy()
            stock["timestamp"] = pd.to_datetime(stock["timestamp"], utc=True)
            stock = stock.sort_values("timestamp").reset_index(drop=True)
            dates = stock["timestamp"].dt.tz_convert("Asia/Kolkata").dt.date.astype(str)
            market_features = calculate_market_features(stock, nifty if not nifty.empty else None, vix if not vix.empty else None)
            trend = calculate_market_trend_regime(nifty) if not nifty.empty else pd.Series(dtype="object")
            volatility = calculate_volatility_regime(vix) if not vix.empty else pd.Series(dtype="object")
            if not nifty.empty:
                trend_frame = pd.DataFrame({"date": nifty["timestamp"].dt.tz_convert("Asia/Kolkata").dt.date.astype(str), "market_trend_regime": trend.to_numpy()})
                market_features["market_trend_regime"] = dates.map(trend_frame.drop_duplicates("date").set_index("date")["market_trend_regime"])
            else:
                market_features["market_trend_regime"] = pd.NA
            if not vix.empty:
                volatility_frame = pd.DataFrame({"date": vix["timestamp"].dt.tz_convert("Asia/Kolkata").dt.date.astype(str), "volatility_regime": volatility.to_numpy()})
                market_features["volatility_regime"] = dates.map(volatility_frame.drop_duplicates("date").set_index("date")["volatility_regime"])
            else:
                market_features["volatility_regime"] = pd.NA
            output = pd.concat(
                [
                    _technical_features(stock),
                    market_features.drop(columns=["date"]),
                    calculate_macro_features(
                        dates,
                        usdinr if not usdinr.empty else None,
                        brent if not brent.empty else None,
                    ),
                ],
                axis=1,
            )
            output["symbol"] = symbol
            output["date"] = dates
            for column in FEATURE_COLUMNS:
                if column not in output:
                    output[column] = pd.NA
            all_rows.extend(output[FEATURE_COLUMNS].to_dict("records"))
        placeholders = ", ".join("?" for _ in FEATURE_COLUMNS)
        rows = [tuple(None if pd.isna(row[column]) else row[column] for column in FEATURE_COLUMNS) for row in all_rows]
        connection.executemany(
            f"INSERT INTO features ({', '.join(FEATURE_COLUMNS)}) VALUES ({placeholders})", rows
        )
        connection.commit()
        return len(rows)


if __name__ == "__main__":
    count = build_features()
    print(f"Built {count} Dataset 2 rows in {DATABASE_PATH}")
    with sqlite3.connect(DATABASE_PATH) as connection:
        for row in connection.execute(
            "SELECT symbol,date,return_20d,rsi_14,volatility_20,nifty_return_20d,india_vix,market_trend_regime "
            "FROM features ORDER BY symbol,date LIMIT 5"
        ):
            print(row)

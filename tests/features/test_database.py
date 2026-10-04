from __future__ import annotations

import sqlite3

import pandas as pd

from scripts.build_features import build_features


def test_feature_table_is_rebuilt_with_primary_key(tmp_path):
    database = tmp_path / "features.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE daily_prices (symbol TEXT, timestamp TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL, open_interest REAL)"
        )
        dates = pd.date_range("2020-01-01", periods=60, freq="D")
        rows = [
            ("TEST", timestamp.isoformat(), 10, 11, 9, index + 1, 100, 0)
            for index, timestamp in enumerate(dates)
        ]
        connection.executemany("INSERT INTO daily_prices VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
        connection.commit()

    assert build_features(database) == 60
    with sqlite3.connect(database) as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(features)")]
        assert "return_20d" in columns
        assert connection.execute("SELECT COUNT(*) FROM features").fetchone()[0] == 60
        assert connection.execute(
            "SELECT COUNT(*) FROM (SELECT symbol, date, COUNT(*) n FROM features GROUP BY symbol, date HAVING n > 1)"
        ).fetchone()[0] == 0

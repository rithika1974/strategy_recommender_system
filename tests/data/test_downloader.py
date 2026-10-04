from __future__ import annotations

import sqlite3

import pandas as pd

import src.data.downloader as downloader


PAYLOAD = {
    "data": {
        "candles": [
            ["2020-01-01T00:00:00+00:00", 100, 110, 90, 105, 1000, 0],
            ["2020-01-02T00:00:00+00:00", 105, 115, 100, 110, 1200, 0],
        ]
    }
}


class FakeClient:
    def get_daily_history(self, instrument_key, from_date, to_date):
        return PAYLOAD


class FakeResolver:
    def __init__(self, **kwargs):
        pass

    def resolve(self, stock):
        if stock["symbol"] == "BROKEN":
            raise ValueError("verified instrument mapping not found")
        return stock


def test_downloader_creates_sqlite_database_and_prevents_duplicates(monkeypatch, tmp_path):
    stocks = [
        {
            "company": "Test Company",
            "symbol": "TEST",
            "category": "large_cap",
            "sector": "Testing",
            "isin": "INE000000000",
            "instrument_key": "NSE_EQ|INE000000000",
        },
        {
            "company": "Broken Company",
            "symbol": "BROKEN",
            "category": "small_cap",
            "sector": "Testing",
            "isin": None,
            "instrument_key": None,
        },
    ]
    monkeypatch.setattr(
        downloader,
        "_load_yaml",
        lambda path: {"universe": {"stocks": stocks, "start_date": "2020-01-01"}}
        if "universe" in str(path)
        else {"start_date": "2020-01-01", "end_date": "2020-01-02"},
    )
    monkeypatch.setattr(downloader, "UpstoxProvider", FakeClient)
    monkeypatch.setattr(downloader, "UpstoxInstrumentResolver", FakeResolver)

    database = tmp_path / "market_data.db"
    first = downloader.download_all_stocks(database_path=database)
    second = downloader.download_all_stocks(database_path=database)

    assert first["results"][0] == {"symbol": "TEST", "records": 2, "status": "downloaded"}
    assert second["results"][0] == {"symbol": "TEST", "records": 0, "status": "up_to_date"}

    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM stocks").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM daily_prices").fetchone()[0] == 2
        assert connection.execute(
            "SELECT company, category, instrument_key, isin FROM stocks WHERE symbol = 'TEST'"
        ).fetchone() == (
            "Test Company",
            "large_cap",
            "NSE_EQ|INE000000000",
            "INE000000000",
        )
        rows = connection.execute(
            "SELECT symbol, timestamp, open, high, low, close, volume, open_interest "
            "FROM daily_prices ORDER BY timestamp"
        ).fetchall()
    assert len(rows) == 2
    assert pd.Timestamp(rows[0][1]).isoformat() == "2020-01-01T00:00:00+00:00"


def test_downloader_can_ingest_only_requested_symbols(monkeypatch, tmp_path):
    stocks = [
        {
            "company": "First Company",
            "symbol": "FIRST",
            "category": "large_cap",
            "sector": "Testing",
            "isin": "INE000000001",
            "instrument_key": "NSE_EQ|INE000000001",
        },
        {
            "company": "Second Company",
            "symbol": "SECOND",
            "category": "large_cap",
            "sector": "Testing",
            "isin": "INE000000002",
            "instrument_key": "NSE_EQ|INE000000002",
        },
    ]
    calls = []
    monkeypatch.setattr(
        downloader,
        "_load_yaml",
        lambda path: {"universe": {"stocks": stocks, "start_date": "2020-01-01"}}
        if "universe" in str(path)
        else {"start_date": "2020-01-01", "end_date": "2020-01-02"},
    )
    monkeypatch.setattr(downloader, "UpstoxProvider", FakeClient)
    monkeypatch.setattr(downloader, "UpstoxInstrumentResolver", FakeResolver)

    class RecordingClient(FakeClient):
        def get_daily_history(self, instrument_key, from_date, to_date):
            calls.append(instrument_key)
            return super().get_daily_history(instrument_key, from_date, to_date)

    monkeypatch.setattr(downloader, "UpstoxProvider", RecordingClient)
    database = tmp_path / "market_data.db"
    with sqlite3.connect(database) as connection:
        downloader.initialize_database(connection)
        connection.execute(
            "INSERT INTO stocks VALUES (?, ?, ?, ?, ?)",
            ("SECOND", "Second Company", "large_cap", "NSE_EQ|INE000000002", "INE000000002"),
        )
        connection.execute(
            "INSERT INTO daily_prices VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("SECOND", "2020-01-01T00:00:00+00:00", 1, 1, 1, 1, 1, 0),
        )
        connection.commit()

    summary = downloader.download_all_stocks(database_path=database, symbols=["FIRST"])

    assert summary["total"] == 1
    assert calls == ["NSE_EQ|INE000000001"]
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM daily_prices WHERE symbol = 'SECOND'"
        ).fetchone()[0] == 1


def test_existing_symbol_downloads_only_missing_period_without_duplicates(monkeypatch, tmp_path):
    stock = {
        "company": "Test Company",
        "symbol": "TEST",
        "category": "large_cap",
        "sector": "Testing",
        "isin": "INE000000000",
        "instrument_key": "NSE_EQ|INE000000000",
    }
    calls = []

    class IncrementalClient:
        def get_daily_history(self, instrument_key, from_date, to_date):
            calls.append((instrument_key, from_date, to_date))
            return {
                "data": {
                    "candles": [
                        ["2020-01-02T00:00:00+00:00", 105, 115, 100, 110, 1200, 0],
                        ["2020-01-03T00:00:00+00:00", 110, 120, 105, 115, 1300, 0],
                    ]
                }
            }

    monkeypatch.setattr(
        downloader,
        "_load_yaml",
        lambda path: {"universe": {"stocks": [stock], "start_date": "2020-01-01"}}
        if "universe" in str(path)
        else {"start_date": "2020-01-01", "end_date": "2020-01-03"},
    )
    monkeypatch.setattr(downloader, "UpstoxProvider", IncrementalClient)
    monkeypatch.setattr(downloader, "UpstoxInstrumentResolver", FakeResolver)
    database = tmp_path / "market_data.db"
    with sqlite3.connect(database) as connection:
        downloader.initialize_database(connection)
        connection.execute(
            "INSERT INTO stocks VALUES (?, ?, ?, ?, ?)",
            ("TEST", "Test Company", "large_cap", "NSE_EQ|INE000000000", "INE000000000"),
        )
        connection.execute(
            "INSERT INTO daily_prices VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("TEST", "2020-01-02T00:00:00+00:00", 105, 115, 100, 110, 1200, 0),
        )
        connection.commit()

    summary = downloader.download_all_stocks(database_path=database)

    assert calls == [("NSE_EQ|INE000000000", "2020-01-03", "2020-01-03")]
    assert summary["results"] == [{"symbol": "TEST", "records": 1, "status": "updated"}]
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM daily_prices WHERE symbol = 'TEST'"
        ).fetchone()[0] == 2
        assert connection.execute(
            "SELECT COUNT(*) FROM (SELECT symbol, timestamp, COUNT(*) n FROM daily_prices "
            "GROUP BY symbol, timestamp HAVING n > 1)"
        ).fetchone()[0] == 0


def test_provider_failure_is_isolated_to_one_stock(monkeypatch, tmp_path):
    stocks = [
        {
            "company": name,
            "symbol": symbol,
            "category": "large_cap",
            "sector": "Testing",
            "isin": isin,
            "instrument_key": f"NSE_EQ|{isin}",
        }
        for name, symbol, isin in (
            ("First Company", "FIRST", "INE000000001"),
            ("Broken Company", "BROKEN_PROVIDER", "INE000000002"),
            ("Last Company", "LAST", "INE000000003"),
        )
    ]

    class PartlyFailingClient:
        def get_daily_history(self, instrument_key, from_date, to_date):
            if instrument_key == "NSE_EQ|INE000000002":
                raise downloader.UpstoxProviderError("temporary provider failure")
            return PAYLOAD

    monkeypatch.setattr(
        downloader,
        "_load_yaml",
        lambda path: {"universe": {"stocks": stocks, "start_date": "2020-01-01"}}
        if "universe" in str(path)
        else {"start_date": "2020-01-01", "end_date": "2020-01-02"},
    )
    monkeypatch.setattr(downloader, "UpstoxProvider", PartlyFailingClient)
    monkeypatch.setattr(downloader, "UpstoxInstrumentResolver", FakeResolver)

    summary = downloader.download_all_stocks(database_path=tmp_path / "market_data.db")

    assert [item["status"] for item in summary["results"]] == [
        "downloaded",
        "failed: temporary provider failure",
        "downloaded",
    ]
    with sqlite3.connect(tmp_path / "market_data.db") as connection:
        assert connection.execute(
            "SELECT DISTINCT symbol FROM daily_prices ORDER BY symbol"
        ).fetchall() == [("FIRST",), ("LAST",)]

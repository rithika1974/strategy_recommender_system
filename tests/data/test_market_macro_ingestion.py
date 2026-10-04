from __future__ import annotations

import sqlite3

import scripts.ingest_market_macro as ingestion


PAYLOAD = {
    "data": {
        "candles": [
            ["2020-01-02T00:00:00+00:00", 100, 110, 90, 105, 1000, 0],
        ]
    }
}


def test_market_instrument_failure_does_not_rollback_other_instruments(monkeypatch):
    class PartlyFailingProvider:
        def get_daily_history(self, instrument_key, from_date, to_date):
            if instrument_key == "BROKEN_KEY":
                raise ValueError("broken market instrument")
            return PAYLOAD

    monkeypatch.setattr(
        ingestion,
        "resolve_indexes",
        lambda: {"NIFTY 50": "NIFTY_KEY", "INDIA VIX": "BROKEN_KEY"},
    )
    monkeypatch.setattr(ingestion, "resolve_global_instruments", lambda: {})
    monkeypatch.setattr(ingestion, "UpstoxProvider", PartlyFailingProvider)

    with sqlite3.connect(":memory:") as connection:
        ingestion.initialize_raw_tables(connection)
        results = ingestion.ingest_indexes(connection)
        connection.commit()

        assert results["NIFTY 50"] == 1
        assert results["INDIA VIX"] == "failed: broken market instrument"
        assert connection.execute(
            "SELECT symbol, COUNT(*) FROM market_data GROUP BY symbol"
        ).fetchall() == [("NIFTY 50", 1)]

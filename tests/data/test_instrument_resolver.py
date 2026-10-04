from __future__ import annotations

import gzip
import json

import pytest

from src.data.instrument_resolver import InstrumentResolutionError, UpstoxInstrumentResolver


class FakeResponse:
    def __init__(self, payload):
        self.content = gzip.compress(json.dumps(payload).encode("utf-8"))

    def raise_for_status(self):
        return None


class FakeSession:
    def __init__(self, payload):
        self.payload = payload

    def get(self, url, timeout):
        return FakeResponse(self.payload)


def test_resolver_verifies_mapping_and_writes_cache(tmp_path):
    master = [
        {
            "segment": "NSE_EQ",
            "instrument_type": "EQ",
            "trading_symbol": "TEST",
            "isin": "INE000000000",
            "instrument_key": "NSE_EQ|INE000000000",
        }
    ]
    resolver = UpstoxInstrumentResolver(
        cache_path=tmp_path / "mapping.json",
        session=FakeSession(master),
    )

    resolved = resolver.resolve(
        {"symbol": "TEST", "isin": "INE000000000", "instrument_key": "NSE_EQ|INE000000000"}
    )

    assert resolved["instrument_key"] == "NSE_EQ|INE000000000"
    assert (tmp_path / "mapping.json").exists()


def test_resolver_rejects_missing_instrument_mapping(tmp_path):
    resolver = UpstoxInstrumentResolver(
        cache_path=tmp_path / "mapping.json",
        session=FakeSession([]),
    )

    with pytest.raises(InstrumentResolutionError, match="expected exactly one"):
        resolver.resolve({"symbol": "UNKNOWN", "isin": None, "instrument_key": None})


def test_resolver_rejects_an_incorrect_configured_key(tmp_path):
    master = [
        {
            "segment": "NSE_EQ",
            "instrument_type": "EQ",
            "trading_symbol": "TEST",
            "isin": "INE000000000",
            "instrument_key": "NSE_EQ|INE000000000",
        }
    ]
    resolver = UpstoxInstrumentResolver(
        cache_path=tmp_path / "mapping.json",
        session=FakeSession(master),
    )

    with pytest.raises(InstrumentResolutionError, match="does not match"):
        resolver.resolve({"symbol": "TEST", "isin": "INE000000000", "instrument_key": "NSE_EQ|WRONG"})

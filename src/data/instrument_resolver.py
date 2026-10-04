from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


INSTRUMENT_MASTER_URL = "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz"


class InstrumentResolutionError(ValueError):
    """Raised when a stock cannot be verified against the Upstox instrument master."""


class UpstoxInstrumentResolver:
    """Resolve and verify NSE equity instruments using Upstox's official master."""

    def __init__(
        self,
        *,
        cache_path: str | Path,
        timeout: int = 60,
        session: requests.Session | None = None,
    ) -> None:
        self.cache_path = Path(cache_path)
        self.timeout = timeout
        self.session = session or requests.Session()
        self._instruments: list[dict[str, Any]] | None = None

    def _load_master(self) -> list[dict[str, Any]]:
        if self._instruments is not None:
            return self._instruments
        response = self.session.get(INSTRUMENT_MASTER_URL, timeout=self.timeout)
        response.raise_for_status()
        master = json.loads(gzip.decompress(response.content))
        if not isinstance(master, list):
            raise InstrumentResolutionError("Upstox instrument master is not a list")
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        compact = {
            "source": INSTRUMENT_MASTER_URL,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "instruments": [
                {
                    "symbol": item.get("trading_symbol"),
                    "isin": item.get("isin"),
                    "instrument_key": item.get("instrument_key"),
                    "segment": item.get("segment"),
                }
                for item in master
                if item.get("segment") == "NSE_EQ" and item.get("instrument_type") in {"EQ", "BE"}
            ],
        }
        for instrument in compact["instruments"]:
            instrument["source"] = compact["source"]
            instrument["retrieved_at"] = compact["retrieved_at"]
        self.cache_path.write_text(json.dumps(compact, indent=2), encoding="utf-8")
        self._instruments = compact["instruments"]
        return self._instruments

    def resolve(self, stock: dict[str, Any]) -> dict[str, Any]:
        symbol = stock.get("symbol")
        if not symbol:
            raise InstrumentResolutionError("Stock is missing symbol")
        instruments = self._load_master()
        matches = [item for item in instruments if item.get("symbol") == symbol]
        if len(matches) != 1:
            raise InstrumentResolutionError(
                f"{symbol}: expected exactly one NSE_EQ mapping in Upstox master, found {len(matches)}"
            )
        resolved = matches[0]
        configured_key = stock.get("instrument_key")
        configured_isin = stock.get("isin")
        if configured_key and configured_key != resolved["instrument_key"]:
            raise InstrumentResolutionError(f"{symbol}: configured instrument_key does not match Upstox master")
        if configured_isin and configured_isin != resolved["isin"]:
            raise InstrumentResolutionError(f"{symbol}: configured ISIN does not match Upstox master")
        return {**stock, "isin": resolved["isin"], "instrument_key": resolved["instrument_key"]}

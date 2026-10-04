from __future__ import annotations

from pathlib import Path

import yaml


UNIVERSE_PATH = Path(__file__).parents[2] / "configs" / "universe.yaml"
REQUIRED_FIELDS = {"company", "symbol", "category", "sector", "isin", "instrument_key"}


def test_development_universe_has_30_entries_and_required_fields():
    config = yaml.safe_load(UNIVERSE_PATH.read_text(encoding="utf-8"))
    stocks = config["universe"]["stocks"]

    assert len(stocks) == 30
    assert {stock["category"] for stock in stocks} == {"large_cap", "mid_cap", "small_cap"}
    assert all(REQUIRED_FIELDS.issubset(stock) for stock in stocks)
    symbols = [stock["symbol"] for stock in stocks]
    assert symbols.count("ALKEM") == 1
    assert "JBCHEPHARM" not in symbols

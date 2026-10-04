"""Market-related schemas."""

from pydantic import BaseModel


class MarketSnapshot(BaseModel):
    """A point-in-time market snapshot."""
    symbol: str
    date: str

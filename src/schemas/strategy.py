"""Strategy-related schemas."""

from pydantic import BaseModel


class StrategyConfig(BaseModel):
    """Strategy configuration."""
    name: str
    params: dict

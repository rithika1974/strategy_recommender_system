"""Recommendation schema."""

from pydantic import BaseModel


class Recommendation(BaseModel):
    """Final recommendation output."""
    stock: str
    recommended_strategy: str | None = None

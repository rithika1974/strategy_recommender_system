"""Risk schemas."""

from pydantic import BaseModel


class RiskAssessment(BaseModel):
    """Risk metadata contract."""
    max_drawdown: float | None = None

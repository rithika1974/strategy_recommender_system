"""Evidence schemas."""

from pydantic import BaseModel


class EvidenceItem(BaseModel):
    """Structured evidence item."""
    source: str
    summary: str

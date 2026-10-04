"""Common input and output contracts for quantitative analysis services."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal


class AnalysisValidationError(ValueError):
    """Raised when a common analysis contract is malformed."""


class AnalysisHorizon(str, Enum):
    """Supported product-level analysis horizons."""

    INTRADAY = "intraday"
    SHORT_TERM = "short_term"
    SWING = "swing"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"


@dataclass(frozen=True)
class AnalysisRequest:
    """One stock and horizon request shared by future system components."""

    stock: str
    horizon: AnalysisHorizon | str

    def __post_init__(self) -> None:
        if not isinstance(self.stock, str):
            raise AnalysisValidationError("stock must be a string")
        symbol = self.stock.strip().upper()
        if not symbol or len(symbol) > 32 or not re.fullmatch(r"[A-Z0-9&-]+", symbol):
            raise AnalysisValidationError("stock must be a valid exchange symbol")
        try:
            horizon = AnalysisHorizon(self.horizon)
        except (TypeError, ValueError) as exc:
            allowed = ", ".join(item.value for item in AnalysisHorizon)
            raise AnalysisValidationError(
                f"unsupported horizon; expected one of: {allowed}"
            ) from exc
        object.__setattr__(self, "stock", symbol)
        object.__setattr__(self, "horizon", horizon)

    def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
        _ = mode
        return {"stock": self.stock, "horizon": self.horizon.value}


@dataclass(frozen=True)
class StrategyOutput:
    """Uniform deterministic evidence emitted by every strategy family."""

    strategy: str
    signal: Literal["long", "short", "flat"]
    parameters: dict[str, Any]
    entry: dict[str, Any] | None
    exit: dict[str, Any] | None
    stop_loss: float | None
    take_profit: float | None
    performance: dict[str, Any]
    risk_metrics: dict[str, Any]
    as_of: str | None
    status: str
    unavailable_fields: list[str]

    def __post_init__(self) -> None:
        if self.signal not in {"long", "short", "flat"}:
            raise AnalysisValidationError(f"unsupported signal: {self.signal}")

    def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
        _ = mode
        return asdict(self)


@dataclass(frozen=True)
class ServiceResponse:
    """JSON-compatible envelope used by all framework-neutral endpoints."""

    status: Literal["ok", "empty", "unavailable"]
    request: dict[str, Any] | None
    source: dict[str, Any]
    data: Any
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.status not in {"ok", "empty", "unavailable"}:
            raise AnalysisValidationError(f"unsupported response status: {self.status}")

    def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
        _ = mode
        return asdict(self)

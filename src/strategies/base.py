"""Shared deterministic signal contract for quantitative strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

import pandas as pd

from src.schemas.analysis import StrategyOutput


Signal = Literal["long", "short", "flat"]
VALID_SIGNALS = {"long", "short", "flat"}


class BaseStrategy(ABC):
    """Base class for stateless, row-level strategy signal rules."""

    name = "base"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}

    def _prepare_features(
        self, features: pd.DataFrame, required_columns: set[str]
    ) -> pd.DataFrame:
        required = {"symbol", "date", *required_columns}
        missing = required.difference(features.columns)
        if missing:
            raise ValueError(
                f"{self.name} requires missing feature columns: {sorted(missing)}"
            )
        return features.sort_values(["symbol", "date"], kind="stable").reset_index(
            drop=True
        )

    def _result(self, features: pd.DataFrame, signals: pd.Series) -> pd.DataFrame:
        result = features[["symbol", "date"]].copy()
        result["strategy"] = self.name
        result["signal"] = signals.astype("string").fillna("flat")
        invalid = set(result["signal"].unique()).difference(VALID_SIGNALS)
        if invalid:
            raise ValueError(f"{self.name} generated invalid signals: {sorted(invalid)}")
        return result

    def structured_output(
        self,
        signal: Signal,
        *,
        as_of: str | None = None,
        performance: dict[str, Any] | None = None,
        risk_metrics: dict[str, Any] | None = None,
    ) -> StrategyOutput:
        """Return the common agent-facing strategy evidence contract.

        The deterministic V1 rules produce direction signals, not executable
        order prices or risk levels. Those unavailable fields remain explicit
        nulls rather than being inferred or fabricated.
        """
        if signal not in VALID_SIGNALS:
            raise ValueError(f"unsupported strategy signal: {signal}")
        return StrategyOutput(
            strategy=self.name,
            signal=signal,
            parameters=dict(self.params),
            entry=None,
            exit=None,
            stop_loss=None,
            take_profit=None,
            performance=dict(performance or {}),
            risk_metrics=dict(risk_metrics or {}),
            as_of=as_of,
            status="signal_only",
            unavailable_fields=["entry", "exit", "stop_loss", "take_profit"],
        )

    @abstractmethod
    def required_feature_columns(self) -> set[str]:
        """Return the existing DB2 columns needed by this strategy."""

    @abstractmethod
    def generate_signals(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return symbol/date/strategy/signal without using future observations."""

"""Deterministic return-threshold momentum strategy."""

from __future__ import annotations

import pandas as pd

from src.strategies.base import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """Go long/short when trailing return crosses a symmetric threshold."""

    name = "momentum"

    def required_feature_columns(self) -> set[str]:
        return {f"return_{int(self.params.get('lookback_days', 20))}d"}

    def generate_signals(self, features: pd.DataFrame) -> pd.DataFrame:
        lookback = int(self.params.get("lookback_days", 20))
        threshold = float(self.params.get("threshold", 0.05))
        if lookback <= 0 or threshold <= 0:
            raise ValueError("momentum lookback_days and threshold must be positive")
        return_column = next(iter(self.required_feature_columns()))
        frame = self._prepare_features(features, self.required_feature_columns())
        values = frame[return_column]
        signals = pd.Series("flat", index=frame.index, dtype="string")
        signals.loc[values.notna() & values.ge(threshold)] = "long"
        signals.loc[values.notna() & values.le(-threshold)] = "short"
        return self._result(frame, signals)

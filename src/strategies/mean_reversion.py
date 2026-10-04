"""Deterministic RSI and Bollinger mean-reversion strategy."""

from __future__ import annotations

import pandas as pd

from src.strategies.base import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """Fade observations that are extreme in both RSI and Bollinger percent-B."""

    name = "mean_reversion"

    def required_feature_columns(self) -> set[str]:
        rsi_window = int(self.params.get("rsi_window", 14))
        return {f"rsi_{rsi_window}", "bollinger_percent_b"}

    def generate_signals(self, features: pd.DataFrame) -> pd.DataFrame:
        rsi_window = int(self.params.get("rsi_window", 14))
        oversold = float(self.params.get("oversold_rsi", 30))
        overbought = float(self.params.get("overbought_rsi", 70))
        lower_band = float(self.params.get("lower_percent_b", 0.0))
        upper_band = float(self.params.get("upper_percent_b", 1.0))
        if rsi_window <= 0 or not 0 <= oversold < overbought <= 100:
            raise ValueError("mean-reversion RSI parameters are invalid")
        if lower_band >= upper_band:
            raise ValueError("lower_percent_b must be below upper_percent_b")
        rsi_column = f"rsi_{rsi_window}"
        frame = self._prepare_features(features, self.required_feature_columns())
        valid = frame[[rsi_column, "bollinger_percent_b"]].notna().all(axis=1)
        signals = pd.Series("flat", index=frame.index, dtype="string")
        signals.loc[
            valid
            & frame[rsi_column].le(oversold)
            & frame["bollinger_percent_b"].le(lower_band)
        ] = "long"
        signals.loc[
            valid
            & frame[rsi_column].ge(overbought)
            & frame["bollinger_percent_b"].ge(upper_band)
        ] = "short"
        return self._result(frame, signals)

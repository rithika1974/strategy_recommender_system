"""Deterministic prior-range breakout strategy."""

from __future__ import annotations

import pandas as pd

from src.strategies.base import BaseStrategy


class BreakoutStrategy(BaseStrategy):
    """Use lookahead-safe prior highs/lows with volume-ratio confirmation."""

    name = "breakout"

    def required_feature_columns(self) -> set[str]:
        lookback = int(self.params.get("lookback_days", 20))
        return {
            "breakout_flag",
            f"distance_from_low_{lookback}",
            f"volume_ratio_{lookback}",
        }

    def generate_signals(self, features: pd.DataFrame) -> pd.DataFrame:
        lookback = int(self.params.get("lookback_days", 20))
        volume_threshold = float(self.params.get("breakout_multiplier", 1.5))
        if lookback <= 0 or volume_threshold < 0:
            raise ValueError("breakout lookback/multiplier parameters are invalid")
        low_distance = f"distance_from_low_{lookback}"
        volume_ratio = f"volume_ratio_{lookback}"
        required = self.required_feature_columns()
        frame = self._prepare_features(features, required)
        valid = frame[list(required)].notna().all(axis=1)
        volume_confirmed = frame[volume_ratio].ge(volume_threshold)
        signals = pd.Series("flat", index=frame.index, dtype="string")
        signals.loc[valid & volume_confirmed & frame["breakout_flag"].eq(1)] = "long"
        signals.loc[
            valid
            & volume_confirmed
            & frame["breakout_flag"].eq(0)
            & frame[low_distance].lt(0)
        ] = "short"
        return self._result(frame, signals)

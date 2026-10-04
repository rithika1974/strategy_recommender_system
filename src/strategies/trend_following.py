"""Deterministic moving-average trend-following strategy."""

from __future__ import annotations

import pandas as pd

from src.strategies.base import BaseStrategy


class TrendFollowingStrategy(BaseStrategy):
    """Confirm price/average direction with the existing market regime."""

    name = "trend_following"

    def required_feature_columns(self) -> set[str]:
        fast = int(self.params.get("fast_window", 20))
        slow = int(self.params.get("slow_window", 50))
        required = {f"price_vs_sma{fast}", f"sma{fast}_vs_sma{slow}"}
        if bool(self.params.get("require_regime_confirmation", True)):
            required.add("market_trend_regime")
        return required

    def generate_signals(self, features: pd.DataFrame) -> pd.DataFrame:
        fast = int(self.params.get("fast_window", 20))
        slow = int(self.params.get("slow_window", 50))
        threshold = float(self.params.get("crossover_threshold", 0.0))
        require_regime = bool(self.params.get("require_regime_confirmation", True))
        if fast <= 0 or slow <= fast or threshold < 0:
            raise ValueError("trend windows/threshold are invalid")
        price_column = f"price_vs_sma{fast}"
        crossover_column = f"sma{fast}_vs_sma{slow}"
        required = self.required_feature_columns()
        frame = self._prepare_features(features, required)
        valid = frame[[price_column, crossover_column]].notna().all(axis=1)
        long_condition = (
            valid
            & frame[price_column].gt(threshold)
            & frame[crossover_column].gt(threshold)
        )
        short_condition = (
            valid
            & frame[price_column].lt(-threshold)
            & frame[crossover_column].lt(-threshold)
        )
        if require_regime:
            long_condition &= frame["market_trend_regime"].eq("BULL")
            short_condition &= frame["market_trend_regime"].eq("BEAR")
        signals = pd.Series("flat", index=frame.index, dtype="string")
        signals.loc[long_condition] = "long"
        signals.loc[short_condition] = "short"
        return self._result(frame, signals)

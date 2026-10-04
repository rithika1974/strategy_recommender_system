from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.strategies.registry import generate_signals_from_database


def main() -> None:
    """Generate V1 strategy signals in memory and print a validation summary."""
    signals, diagnostics = generate_signals_from_database()
    distribution = (
        signals.groupby(["strategy", "signal"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["long", "short", "flat"], fill_value=0)
    )
    summary = {
        "symbols": int(signals["symbol"].nunique()),
        "rows_per_strategy": int(
            signals.groupby("strategy").size().iloc[0] if not signals.empty else 0
        ),
        "signal_distribution": {
            strategy: {signal: int(count) for signal, count in counts.items()}
            for strategy, counts in distribution.to_dict(orient="index").items()
        },
        "diagnostics": diagnostics,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

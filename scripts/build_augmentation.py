from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.augmentation.rolling_windows import build_augmentation_windows


def main() -> None:
    """Build compact rolling-window references for DB2 features."""
    summary = build_augmentation_windows()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.downloader import download_all_stocks

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the Dataset 1 download flow."""
    summary = download_all_stocks()

    successful = 0
    for item in summary["results"]:
        logger.info("%s | %s | %s", item["symbol"], item["records"], item["status"])
        if not item["status"].startswith("failed:"):
            successful += 1

    logger.info("Total stocks: %s", summary["total"])
    logger.info("Downloaded successfully: %s", successful)
    logger.info("Failed/unresolved: %s", summary["total"] - successful)


if __name__ == "__main__":
    main()

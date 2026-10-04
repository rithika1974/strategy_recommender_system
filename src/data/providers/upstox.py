from __future__ import annotations

import logging
import os
import time
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class UpstoxProviderError(RuntimeError):
    """Raised when the Upstox provider encounters a request or parsing problem."""


class UpstoxAuthenticationError(UpstoxProviderError):
    """Raised when the token or auth configuration is invalid."""


class UpstoxProvider:
    """Minimal provider for the official Upstox historical candle API."""

    BASE_URL = "https://api.upstox.com"
    API_VERSION = "v3"

    def __init__(
        self,
        access_token: str | None = None,
        timeout: int = 30,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
    ) -> None:
        self.access_token = access_token or os.getenv("UPSTOX_ACCESS_TOKEN")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        if self.access_token:
            self.session.headers.update({"Authorization": "Bearer " + self.access_token})

    def _build_url(self, instrument_key: str, from_date: str, to_date: str) -> str:
        if not instrument_key:
            raise ValueError("instrument_key must not be empty")
        from_date = str(from_date).strip()
        to_date = str(to_date).strip()
        if not from_date or not to_date:
            raise ValueError("from_date and to_date must be provided")
        return (
            f"{self.BASE_URL}/{self.API_VERSION}/historical-candle/"
            f"{instrument_key}/days/1/{to_date}/{from_date}"
        )

    def get_daily_history(self, instrument_key: str, from_date: str, to_date: str) -> dict[str, Any]:
        """Return the raw JSON payload for daily historical candles."""
        if not self.access_token:
            raise UpstoxAuthenticationError("UPSTOX_ACCESS_TOKEN is missing from the environment")

        url = self._build_url(instrument_key, from_date, to_date)
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(url, timeout=self.timeout)
            except requests.Timeout as exc:  # pragma: no cover - network-specific
                if attempt < self.max_retries:
                    logger.warning(
                        "Upstox timeout for %s; retrying (%s/%s)",
                        instrument_key,
                        attempt,
                        self.max_retries,
                    )
                    time.sleep(self.backoff_seconds * (2 ** (attempt - 1)))
                    continue
                raise UpstoxProviderError(f"Upstox request timed out for {instrument_key}") from exc
            except requests.RequestException as exc:  # pragma: no cover - network-specific
                raise UpstoxProviderError(f"Upstox request failed for {instrument_key}: {exc}") from exc

            if response.status_code in (401, 403):
                raise UpstoxAuthenticationError(
                    "Upstox rejected the provided access token; confirm the token is valid and active"
                )
            if response.status_code in (429, 500, 502, 503, 504):
                if attempt < self.max_retries:
                    logger.warning(
                        "Upstox transient error %s for %s; retrying (%s/%s)",
                        response.status_code,
                        instrument_key,
                        attempt,
                        self.max_retries,
                    )
                    time.sleep(self.backoff_seconds * (2 ** (attempt - 1)))
                    continue
            if response.status_code != 200:
                raise UpstoxProviderError(
                    f"Upstox returned HTTP {response.status_code} for {instrument_key}"
                )
            try:
                return response.json()
            except ValueError as exc:
                raise UpstoxProviderError(f"Upstox returned malformed JSON for {instrument_key}") from exc

        raise UpstoxProviderError(
            f"Upstox request failed after {self.max_retries} attempts for {instrument_key}"
        )

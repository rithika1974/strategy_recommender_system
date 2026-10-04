from __future__ import annotations

from types import SimpleNamespace

from src.data.providers.upstox import UpstoxProvider


class RetrySession:
    def __init__(self):
        self.headers = {}
        self.calls = 0

    def get(self, url, timeout):
        self.calls += 1
        if self.calls < 3:
            return SimpleNamespace(status_code=503)
        return SimpleNamespace(status_code=200, json=lambda: {"data": {"candles": []}})


def test_provider_retries_transient_failures_with_exponential_backoff(monkeypatch):
    provider = UpstoxProvider(access_token="test-token", max_retries=3, backoff_seconds=0)
    session = RetrySession()
    provider.session = session
    sleeps = []
    monkeypatch.setattr("src.data.providers.upstox.time.sleep", sleeps.append)

    provider.get_daily_history("NSE_EQ|TEST", "2020-01-01", "2020-01-02")

    assert session.calls == 3
    assert sleeps == [0, 0]

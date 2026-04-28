"""Tests for Zerodha quote_bulk index alias handling."""

from __future__ import annotations

import os
from typing import Any

import pytest

# Set credential env BEFORE importing the client so __init__ does not raise.
os.environ.setdefault("ZERODHA_API_KEY", "test_api_key")
os.environ.setdefault("ZERODHA_ACCESS_TOKEN", "test_access_token")

from nifty_scalper_bot.data.rest.zerodha_client import ZerodhaKiteClient


class _FakeRateLimiter:
    """Minimal rate limiter that bypasses real bucket acquisition."""

    def __init__(self) -> None:
        self.acquire_calls: list[tuple[str, int]] = []

    def add_bucket(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def acquire(self, bucket: str, weight: int = 1) -> None:
        self.acquire_calls.append((bucket, weight))


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> ZerodhaKiteClient:
    monkeypatch.setenv("ZERODHA_API_KEY", "test_api_key")
    monkeypatch.setenv("ZERODHA_ACCESS_TOKEN", "test_access_token")
    return ZerodhaKiteClient(limiter=_FakeRateLimiter())


def _capture(client: ZerodhaKiteClient, response_data: dict[str, Any]) -> list[dict]:
    captured: list[dict] = []

    def fake_request(
        method: str,
        endpoint: str,
        params: dict | None = None,
        data: dict | None = None,
        **_kwargs: Any,
    ) -> dict:
        captured.append({"method": method, "endpoint": endpoint, "params": params})
        return {"data": response_data}

    client._make_request = fake_request  # type: ignore[assignment]
    client._acquire_bucket = lambda *_a, **_k: None  # type: ignore[assignment]
    return captured


def test_get_quote_bulk_translates_nifty_alias_and_returns_under_original_key(
    client: ZerodhaKiteClient,
) -> None:
    captured = _capture(
        client,
        {"NSE:NIFTY 50": {"last_price": 24027.3, "instrument_token": 256265}},
    )

    result = client.get_quote_bulk(["NSE:NIFTY"])

    # Caller-facing key remains "NSE:NIFTY".
    assert "NSE:NIFTY" in result
    assert result["NSE:NIFTY"]["last_price"] == pytest.approx(24027.3)
    # The Kite alias must have been sent to /quote.
    assert captured, "No HTTP call captured"
    sent = captured[0]["params"]["i"]
    assert "NSE:NIFTY 50" in sent


def test_quote_alias_for_nifty_returns_payload_under_original_key(
    client: ZerodhaKiteClient,
) -> None:
    _capture(
        client,
        {"NSE:NIFTY 50": {"last_price": 24500.0, "instrument_token": 256265}},
    )

    result = client.quote(["NSE:NIFTY"])
    assert "NSE:NIFTY" in result
    assert result["NSE:NIFTY"]["last_price"] == pytest.approx(24500.0)


def test_quote_bulk_passes_through_non_index_symbol(
    client: ZerodhaKiteClient,
) -> None:
    captured = _capture(
        client,
        {"NFO:NIFTY2612025700CE": {"last_price": 102.0, "instrument_token": 999}},
    )

    result = client.get_quote_bulk(["NFO:NIFTY2612025700CE"])

    assert "NFO:NIFTY2612025700CE" in result
    assert result["NFO:NIFTY2612025700CE"]["last_price"] == pytest.approx(102.0)
    sent = captured[0]["params"]["i"]
    assert "NFO:NIFTY2612025700CE" in sent
    # No index alias rewriting should leak.
    assert "NSE:NIFTY 50" not in sent


def test_get_quote_bulk_marks_quote_api_unavailable_on_403(
    client: ZerodhaKiteClient,
) -> None:
    def fake_request(*_args: Any, **_kwargs: Any) -> dict:
        raise RuntimeError("Zerodha API error (403): HTTP 403 access denied")

    client._make_request = fake_request  # type: ignore[assignment]
    client._acquire_bucket = lambda *_a, **_k: None  # type: ignore[assignment]

    result = client.get_quote_bulk(["NSE:NIFTY"])
    assert result == {}
    snap = client.quote_api_status_snapshot()
    assert snap["available"] is False
    assert snap["error"] == "access_denied"

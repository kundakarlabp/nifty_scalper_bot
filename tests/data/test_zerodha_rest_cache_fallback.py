"""Tests for Zerodha REST cache fallback behavior."""

from __future__ import annotations

from typing import Any

import pytest

from nifty_scalper_bot.data.rest.zerodha_client import (
    ZerodhaKiteClient,
    _RestCacheEntry,
)
from nifty_scalper_bot.utils.errors import BrokerError


def _build_client() -> ZerodhaKiteClient:
    """Create Zerodha client with minimal credentials."""

    return ZerodhaKiteClient(api_key="key", access_token="token")


def test_get_orders_returns_cached_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use cached orders when REST fetch fails."""

    client = _build_client()
    now = 1_500.0
    monkeypatch.setattr(client, "_log_time_fn", lambda: now)
    client._rest_cache_ttl = 30.0
    client._orders_cache = _RestCacheEntry(
        payload=[{"order_id": "abc123"}],
        updated_at=now - 5.0,
    )

    def _raise(*_: Any, **__: Any) -> None:
        raise BrokerError("boom")

    monkeypatch.setattr(client, "_execute_with_retry", _raise)
    result = client.get_orders()

    assert result == [{"order_id": "abc123"}]


def test_get_positions_returns_cached_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use cached positions when REST fetch fails."""

    client = _build_client()
    now = 1_000.0
    monkeypatch.setattr(client, "_log_time_fn", lambda: now)
    client._rest_cache_ttl = 30.0
    client._positions_cache = _RestCacheEntry(
        payload=[{"tradingsymbol": "NIFTY"}],
        updated_at=now - 5.0,
    )

    def _raise(*_: Any, **__: Any) -> None:
        raise BrokerError("boom")

    monkeypatch.setattr(client, "_execute_with_retry", _raise)
    result = client.get_positions()

    assert result == [{"tradingsymbol": "NIFTY"}]


def test_get_account_margins_returns_cached_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use cached margin payload when REST fetch fails."""

    client = _build_client()
    now = 2_000.0
    monkeypatch.setattr(client, "_log_time_fn", lambda: now)
    client._rest_cache_ttl = 30.0
    client._margins_cache["equity"] = _RestCacheEntry(
        payload={"available": {"cash": 125_000.0}},
        updated_at=now - 4.0,
    )

    def _raise(*_: Any, **__: Any) -> None:
        raise BrokerError("boom")

    monkeypatch.setattr(client, "_execute_with_retry", _raise)
    result = client.get_account_margins(segment="equity")

    assert result == {"available": {"cash": 125_000.0}}

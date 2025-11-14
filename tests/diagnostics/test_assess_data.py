"""Tests for data layer assessment helpers."""

from __future__ import annotations

import time
from typing import Any, Callable

import pytest

from nifty_scalper_bot.data.assess_data import assess_datahub_fresh
from nifty_scalper_bot.data.data_hub import DataHub


class _StubMarketDataManager:
    def __init__(self) -> None:
        self._latest: dict[str, dict[str, Any]] = {}

    def subscribe(
        self, symbol: str, callback: Callable[[dict[str, Any]], None]
    ) -> None:
        self._latest.setdefault(symbol, {})

    def unsubscribe(
        self, symbol: str, callback: Callable[[dict[str, Any]], None]
    ) -> None:
        self._latest.pop(symbol, None)

    def get_latest_tick(self, symbol: str) -> dict[str, Any] | None:
        return self._latest.get(symbol)

    def pull_quote(self, symbol: str) -> dict[str, Any] | None:
        return self._latest.get(symbol)


class _StubResolver:
    def __getattr__(self, _name: str) -> Any:  # pragma: no cover - optional hooks
        raise AttributeError(_name)


@pytest.fixture()
def stub_mdm() -> _StubMarketDataManager:
    return _StubMarketDataManager()


@pytest.fixture()
def resolver() -> _StubResolver:
    return _StubResolver()


def test_assess_datahub_fresh_accepts_within_adaptive(stub_mdm, resolver) -> None:
    hub = DataHub(stub_mdm, resolver, options_only=True)
    now = time.time()
    hub._now = lambda: now  # type: ignore[attr-defined]
    symbol = "NIFTY"
    hub.ingest_tick({"symbol": symbol, "timestamp": now - 1.2})
    ok, detail, meta = assess_datahub_fresh(hub, symbol, adaptive_ms=1750)
    assert ok
    assert detail == "ok"
    assert meta["age_ms"] <= 1750

"""Regression tests for mandatory Zerodha market protection on market orders.

Since 2026-04-01 Zerodha rejects API MARKET/SL-M orders that do not carry a
``market_protection`` value (HTTP 400). ``ZerodhaKiteClient.place_order`` must
inject it automatically (default ``-1`` = automatic protection) while leaving
LIMIT / SL orders untouched.
"""

from __future__ import annotations

from typing import Any

import pytest

from nifty_scalper_bot.data.rest.zerodha_client import ZerodhaKiteClient


def _make_client(monkeypatch: pytest.MonkeyPatch) -> tuple[ZerodhaKiteClient, dict]:
    client = ZerodhaKiteClient(api_key="k", access_token="t")
    captured: dict[str, Any] = {}

    def _fake_make_request(*args: Any, **kwargs: Any) -> dict[str, Any]:
        captured["data"] = kwargs.get("data")
        captured["path"] = args[1] if len(args) > 1 else kwargs.get("path")
        return {"status": "success", "data": {"order_id": "ORD123"}}

    monkeypatch.setattr(client, "_make_request", _fake_make_request)
    monkeypatch.setattr(client, "_acquire_bucket", lambda *a, **k: None)
    return client, captured


async def test_market_order_injects_default_market_protection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ZERODHA_MARKET_PROTECTION", raising=False)
    client, captured = _make_client(monkeypatch)

    res = client.place_order(
        symbol="NFO:NIFTY2662324150CE",
        side="SELL",
        quantity=65,
        order_type="MARKET",
        product="MIS",
    )

    assert res["order_id"] == "ORD123"
    assert captured["data"]["order_type"] == "MARKET"
    # -1 => automatic market protection applied by Zerodha per exchange band.
    assert captured["data"]["market_protection"] == -1


async def test_slm_order_injects_market_protection_and_keeps_trigger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ZERODHA_MARKET_PROTECTION", raising=False)
    client, captured = _make_client(monkeypatch)

    client.place_order(
        symbol="NFO:NIFTY2662324150CE",
        side="SELL",
        quantity=65,
        order_type="STOP_LOSS_MARKET",
        trigger_price=120.0,
        product="MIS",
    )

    assert captured["data"]["order_type"] == "SL-M"
    assert captured["data"]["market_protection"] == -1
    # SL-M must retain its trigger price (only plain MARKET drops it).
    assert captured["data"]["trigger_price"] == 120.0


async def test_limit_order_has_no_market_protection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, captured = _make_client(monkeypatch)

    client.place_order(
        symbol="NFO:NIFTY2662324150CE",
        side="BUY",
        quantity=65,
        order_type="LIMIT",
        price=130.0,
        product="MIS",
    )

    assert captured["data"]["order_type"] == "LIMIT"
    assert "market_protection" not in captured["data"]


async def test_market_protection_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ZERODHA_MARKET_PROTECTION", "5")
    client, captured = _make_client(monkeypatch)

    client.place_order(
        symbol="NFO:NIFTY2662324150CE",
        side="SELL",
        quantity=65,
        order_type="MARKET",
        product="MIS",
    )

    assert captured["data"]["market_protection"] == 5


async def test_explicit_market_protection_is_respected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ZERODHA_MARKET_PROTECTION", raising=False)
    client, captured = _make_client(monkeypatch)

    client.place_order(
        symbol="NFO:NIFTY2662324150CE",
        side="SELL",
        quantity=65,
        order_type="MARKET",
        product="MIS",
        market_protection=10,
    )

    # Caller-supplied value must win over the default (setdefault semantics).
    assert captured["data"]["market_protection"] == 10

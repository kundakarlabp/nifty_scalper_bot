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


async def test_limit_order_removes_supplied_market_protection(
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
        market_protection=5,
    )

    assert captured["data"]["order_type"] == "LIMIT"
    assert "market_protection" not in captured["data"]


async def test_sl_order_removes_supplied_market_protection_and_keeps_prices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, captured = _make_client(monkeypatch)

    client.place_order(
        symbol="NFO:NIFTY2662324150CE",
        side="SELL",
        quantity=65,
        order_type="SL",
        price=120.0,
        trigger_price=121.0,
        product="MIS",
        market_protection=5,
    )

    assert captured["data"]["order_type"] == "SL"
    assert captured["data"]["price"] == 120.0
    assert captured["data"]["trigger_price"] == 121.0
    assert "market_protection" not in captured["data"]


async def test_market_order_zero_market_protection_is_rejected_before_broker_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, captured = _make_client(monkeypatch)

    with pytest.raises(Exception, match="market_protection must be -1"):
        client.place_order(
            symbol="NFO:NIFTY2662324150CE",
            side="SELL",
            quantity=65,
            order_type="MARKET",
            product="MIS",
            market_protection=0,
        )

    assert "data" not in captured


async def test_invalid_order_type_rejects_before_broker_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, captured = _make_client(monkeypatch)

    with pytest.raises(Exception, match="Unsupported Zerodha order_type"):
        client.place_order(
            symbol="NFO:NIFTY2662324150CE",
            side="BUY",
            quantity=65,
            order_type="BOGUS",
            product="MIS",
        )

    assert "data" not in captured


async def test_acknowledgement_is_submitted_not_success_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _captured = _make_client(monkeypatch)

    result = client.place_order(
        symbol="NFO:NIFTY2662324150CE",
        side="BUY",
        quantity=65,
        order_type="MARKET",
        product="MIS",
    )

    assert result["submitted"] is True
    assert result["status"] == "SUBMITTED"
    assert result["raw_status"] == "success"


async def test_acknowledgement_without_order_id_is_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = ZerodhaKiteClient(api_key="k", access_token="t")
    calls = {"count": 0}

    def _fake_make_request(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls["count"] += 1
        return {"status": "success", "data": {}}

    monkeypatch.setattr(client, "_make_request", _fake_make_request)
    monkeypatch.setattr(client, "_acquire_bucket", lambda *a, **k: None)

    with pytest.raises(Exception, match="order_id"):
        client.place_order(
            symbol="NFO:NIFTY2662324150CE",
            side="BUY",
            quantity=65,
            order_type="MARKET",
            product="MIS",
        )

    assert calls["count"] == 1


@pytest.mark.parametrize("value", [-2, 0, 101, "invalid", 1.5])
async def test_market_order_invalid_market_protection_rejected_before_broker_call(
    monkeypatch: pytest.MonkeyPatch,
    value: Any,
) -> None:
    client, captured = _make_client(monkeypatch)

    with pytest.raises(Exception, match="market_protection must be -1"):
        client.place_order(
            symbol="NFO:NIFTY2662324150CE",
            side="SELL",
            quantity=65,
            order_type="MARKET",
            product="MIS",
            market_protection=value,
        )

    assert "data" not in captured


@pytest.mark.parametrize("value", [-1, 1, 10, 100])
async def test_market_order_valid_market_protection_values_are_sent(
    monkeypatch: pytest.MonkeyPatch,
    value: int,
) -> None:
    client, captured = _make_client(monkeypatch)

    client.place_order(
        symbol="NFO:NIFTY2662324150CE",
        side="SELL",
        quantity=65,
        order_type="MARKET",
        product="MIS",
        market_protection=value,
    )

    assert captured["data"]["market_protection"] == value


async def test_market_order_removes_price_and_trigger_price(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, captured = _make_client(monkeypatch)

    client.place_order(
        symbol="NFO:NIFTY2662324150CE",
        side="BUY",
        quantity=65,
        order_type="MARKET",
        price=123.0,
        trigger_price=122.0,
        product="MIS",
    )

    assert "price" not in captured["data"]
    assert "trigger_price" not in captured["data"]

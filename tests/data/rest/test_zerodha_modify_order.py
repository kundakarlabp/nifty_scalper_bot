"""Broker-adapter regression tests for Zerodha order modification semantics."""

from __future__ import annotations

from typing import Any

import pytest

from nifty_scalper_bot.data.rest.zerodha_client import ZerodhaKiteClient


def _client(monkeypatch: pytest.MonkeyPatch) -> tuple[ZerodhaKiteClient, dict[str, Any]]:
    client = ZerodhaKiteClient(api_key="k", access_token="t")
    captured: dict[str, Any] = {}

    def _request(*args: Any, **kwargs: Any) -> dict[str, Any]:
        captured["method"] = args[0]
        captured["path"] = args[1]
        captured["data"] = kwargs.get("data")
        return {"status": "success", "data": {"order_id": "ORD1"}}

    monkeypatch.setattr(client, "_make_request", _request)
    monkeypatch.setattr(client, "_acquire_bucket", lambda *args, **kwargs: None)
    return client, captured


def test_limit_price_modify_does_not_invent_trigger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, captured = _client(monkeypatch)

    client.modify_order("ORD1", price=101.25)

    assert captured["data"] == {"price": 101.25}


def test_slm_trigger_modify_does_not_invent_limit_price(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, captured = _client(monkeypatch)

    client.modify_order("ORD1", trigger_price=99.5)

    assert captured["data"] == {"trigger_price": 99.5}


def test_sl_modify_preserves_distinct_price_and_trigger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, captured = _client(monkeypatch)

    client.modify_order("ORD1", price=98.0, trigger_price=99.0)

    assert captured["data"] == {"price": 98.0, "trigger_price": 99.0}


def test_quantity_modify_does_not_change_price_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, captured = _client(monkeypatch)

    client.modify_order("ORD1", quantity=65)

    assert captured["data"] == {"quantity": 65}


def test_modify_keeps_existing_positional_variety_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, captured = _client(monkeypatch)

    client.modify_order("ORD1", 65, 101.0, "amo")

    assert captured["path"] == "/orders/amo/ORD1"
    assert captured["data"] == {"quantity": 65, "price": 101.0}


def test_empty_modify_is_rejected_before_broker_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, captured = _client(monkeypatch)

    with pytest.raises(Exception, match="quantity, price, or trigger_price"):
        client.modify_order("ORD1")

    assert "data" not in captured

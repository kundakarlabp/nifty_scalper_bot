from __future__ import annotations

from typing import Any

import pytest

from nifty_scalper_bot.data.rest.zerodha_client import ZerodhaKiteClient
from nifty_scalper_bot.utils.errors import BrokerError


def _client(monkeypatch: pytest.MonkeyPatch) -> tuple[ZerodhaKiteClient, dict[str, Any]]:
    client = ZerodhaKiteClient(api_key="k", api_secret="s", access_token="t")
    captured: dict[str, Any] = {}

    def _request(*args: Any, **kwargs: Any) -> dict[str, Any]:
        captured["data"] = kwargs.get("data")
        return {"status": "success", "data": {"order_id": "OID-1"}}

    monkeypatch.setattr(client, "_make_request", _request)
    monkeypatch.setattr(client, "_acquire_bucket", lambda *args, **kwargs: None)
    return client, captured


@pytest.mark.parametrize(
    ("symbol", "message"),
    [
        ("NFO:NIFTY25APR25000FUT", "Futures disabled"),
        ("NFO:NIFTY25APR25000XX", "Only NIFTY options"),
        ("NSE:NIFTY25APR25000CE", "Only NFO exchange"),
    ],
)
def test_place_order_rejects_non_executable_contracts_before_broker_call(
    monkeypatch: pytest.MonkeyPatch,
    symbol: str,
    message: str,
) -> None:
    client, captured = _client(monkeypatch)

    with pytest.raises(BrokerError, match=message):
        client.place_order(
            symbol=symbol,
            side="BUY",
            quantity=75,
            order_type="MARKET",
        )

    assert "data" not in captured


def test_place_order_preserves_tradingsymbol_case(monkeypatch: pytest.MonkeyPatch) -> None:
    client, captured = _client(monkeypatch)

    result = client.place_order(
        symbol="nfo:nifty25apr25000ce",
        side="BUY",
        quantity=75,
        order_type="LIMIT",
        price=100.0,
    )

    assert result["order_id"] == "OID-1"
    assert captured["data"]["tradingsymbol"] == "nifty25apr25000ce"
    assert captured["data"]["exchange"] == "NFO"


@pytest.mark.parametrize("side", ["", "HOLD", "LONG"])
def test_place_order_rejects_invalid_side_before_broker_call(
    monkeypatch: pytest.MonkeyPatch,
    side: str,
) -> None:
    client, captured = _client(monkeypatch)

    with pytest.raises(BrokerError, match="Missing side"):
        client.place_order(
            symbol="NFO:NIFTY25APR25000CE",
            side=side,
            quantity=75,
            order_type="MARKET",
        )

    assert "data" not in captured


@pytest.mark.parametrize("quantity", [0, -75, "bad"])
def test_place_order_rejects_invalid_quantity_before_broker_call(
    monkeypatch: pytest.MonkeyPatch,
    quantity: Any,
) -> None:
    client, captured = _client(monkeypatch)

    with pytest.raises(BrokerError, match="Invalid quantity"):
        client.place_order(
            symbol="NFO:NIFTY25APR25000CE",
            side="BUY",
            quantity=quantity,
            order_type="MARKET",
        )

    assert "data" not in captured

"""Tests for Zerodha client option-only guardrails."""

import pytest

from nifty_scalper_bot.data.rest.zerodha_client import ZerodhaKiteClient
from nifty_scalper_bot.utils.errors import BrokerError


class _ResolverStub:
    """Stub resolver returning preconfigured exchange + symbol."""

    def __init__(self, tradingsymbol: str) -> None:
        self._tradingsymbol = tradingsymbol

    @staticmethod
    def exchange_for_symbol(_symbol: str) -> str:
        return "NFO"

    def tradingsymbol_for_order(self, _symbol: str) -> str:
        return self._tradingsymbol


def _make_client(resolver: _ResolverStub | None = None) -> ZerodhaKiteClient:
    client = ZerodhaKiteClient(api_key="key", access_token="token")
    if resolver is not None:
        client.attach_resolver(resolver)
    return client


def test_build_kite_params_rejects_futures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INSTRUMENTS__ALLOW_FUTURES", raising=False)
    client = _make_client(_ResolverStub("NIFTY25OCTFUT"))

    payload = {
        "symbol": "NIFTY",
        "side": "BUY",
        "quantity": 25,
        "order_type": "MARKET",
    }

    with pytest.raises(BrokerError, match="Futures disabled"):
        client._build_kite_params(payload)

    client._client.close()


def test_build_kite_params_requires_option_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INSTRUMENTS__ALLOW_FUTURES", raising=False)
    client = _make_client(_ResolverStub("NIFTY"))

    payload = {
        "symbol": "NIFTY",
        "side": "SELL",
        "quantity": 25,
        "order_type": "MARKET",
    }

    with pytest.raises(BrokerError, match="Only NIFTY options"):
        client._build_kite_params(payload)

    client._client.close()


def test_build_kite_params_accepts_option(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INSTRUMENTS__ALLOW_FUTURES", raising=False)
    resolver = _ResolverStub("NIFTY25OCT20000CE")
    client = _make_client(resolver)

    payload = {
        "symbol": "NIFTY",
        "side": "BUY",
        "quantity": 50,
        "order_type": "MARKET",
    }

    params = client._build_kite_params(payload)

    assert params["tradingsymbol"] == "NIFTY25OCT20000CE"
    assert params["exchange"] == "NFO"

    client._client.close()

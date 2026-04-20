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


def test_get_quote_accepts_alternate_nifty_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client()

    def fake_request(*_args, **_kwargs):  # noqa: ANN001
        return {
            "data": {
                "NSE:NIFTY 50": {
                    "last_price": 25234.5,
                    "depth": {"buy": [], "sell": []},
                }
            }
        }

    monkeypatch.setattr(client, "_make_request", fake_request)
    monkeypatch.setattr(client, "_ensure_json", lambda payload: payload)

    quote = client.get_quote("NSE:NIFTY")

    assert quote["symbol"] == "NSE:NIFTY"
    assert quote["ltp"] == pytest.approx(25234.5)
    client._client.close()


def test_get_quote_sends_all_nifty_aliases_in_single_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``get_quote`` must ask Kite for every known NIFTY alias in one call.

    Kite rejects the bare ``NSE:NIFTY`` key for the spot index; previously
    the client only sent the caller-facing alias which produced an empty
    payload and the caller observed ``Quote data missing for NSE:NIFTY``.
    """

    client = _make_client()
    captured: dict[str, list[str]] = {}

    def fake_request(method: str, path: str, params=None, **_kwargs):  # noqa: ANN001
        captured["method"] = method
        captured["path"] = path
        captured["i"] = list((params or {}).get("i", []))
        return {
            "data": {
                "NSE:NIFTY 50": {
                    "last_price": 25110.75,
                    "depth": {"buy": [], "sell": []},
                }
            }
        }

    monkeypatch.setattr(client, "_make_request", fake_request)
    monkeypatch.setattr(client, "_ensure_json", lambda payload: payload)

    quote = client.get_quote("NSE:NIFTY")

    assert quote["ltp"] == pytest.approx(25110.75)
    assert "NSE:NIFTY 50" in captured["i"], captured
    assert "NIFTY 50" in captured["i"], captured
    client._client.close()


def test_get_quote_falls_back_to_first_payload_when_keys_are_unexpected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kite occasionally keys the payload by instrument token.

    A non-empty mapping must still resolve to a quote instead of raising
    ``Quote data missing`` — the payload itself is authoritative.
    """

    client = _make_client()

    def fake_request(*_args, **_kwargs):  # noqa: ANN001
        return {
            "data": {
                "256265": {
                    "last_price": 25222.15,
                    "depth": {"buy": [], "sell": []},
                }
            }
        }

    monkeypatch.setattr(client, "_make_request", fake_request)
    monkeypatch.setattr(client, "_ensure_json", lambda payload: payload)

    quote = client.get_quote("NSE:NIFTY")
    assert quote["ltp"] == pytest.approx(25222.15)
    client._client.close()

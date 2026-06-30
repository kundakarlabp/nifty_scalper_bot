"""Tests for Zerodha client option-only guardrails."""

from datetime import datetime, timedelta, timezone

import httpx
import pytest

from nifty_scalper_bot.data.rest.zerodha_client import (
    ZerodhaKiteClient,
    _format_kite_datetime,
)
from nifty_scalper_bot.utils.errors import (
    BrokerError,
    ConfigurationError,
    OrderPlacementError,
)


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


def test_quote_403_marks_quote_api_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client()

    def fail_request(*_args, **_kwargs):  # noqa: ANN001
        raise RuntimeError("HTTP 403 access denied")

    monkeypatch.setattr(client, "_make_request", fail_request)

    assert client.quote_api_available() is True
    result = client.quote_any(["NSE:NIFTY"])
    assert result is None
    status = client.quote_api_status_snapshot()
    assert status["available"] is False
    assert status["error"] == "access_denied"
    assert isinstance(status["last_checked_at"], float)
    client._client.close()


def test_margins_success_does_not_override_quote_api_denied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client()
    client._mark_quote_api_status(available=False, error="access_denied")

    def fake_margins(*_args, **_kwargs):  # noqa: ANN001
        return {"equity": {"available": {"cash": 1000.0}}}

    monkeypatch.setattr(client, "_make_request", fake_margins)
    payload = client.get_margins(segment="equity")
    assert payload["equity"]["available"]["cash"] == pytest.approx(1000.0)
    assert client.quote_api_available() is False
    status = client.quote_api_status_snapshot()
    assert status["error"] == "access_denied"
    client._client.close()


def test_historical_data_accepts_positive_integer_and_serializes_utc_to_ist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client()
    captured = {}

    def fake_request(method: str, path: str, params=None, **_kwargs):  # noqa: ANN001
        captured.update(method=method, path=path, params=dict(params or {}))
        return {"data": {"candles": []}}

    monkeypatch.setattr(client, "_make_request", fake_request)
    monkeypatch.setattr(client, "_ensure_json", lambda payload: payload)
    monkeypatch.setattr(client, "_acquire_bucket", lambda _bucket: None)

    rows = client.historical_data(
        12345,
        datetime(2026, 6, 29, 6, 25, tzinfo=timezone.utc),
        datetime(2026, 6, 29, 6, 30, tzinfo=timezone.utc),
        "minute",
    )

    assert rows == []
    assert captured["path"] == "/instruments/historical/12345/minute"
    assert captured["params"]["from"] == "2026-06-29 11:55:00"
    assert captured["params"]["to"] == "2026-06-29 12:00:00"
    client._client.close()


def test_historical_data_accepts_numeric_string_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client()
    captured = {}

    def fake_request(_method: str, path: str, params=None, **_kwargs):  # noqa: ANN001
        captured["path"] = path
        return {"data": {"candles": []}}

    monkeypatch.setattr(client, "_make_request", fake_request)
    monkeypatch.setattr(client, "_ensure_json", lambda payload: payload)
    monkeypatch.setattr(client, "_acquire_bucket", lambda _bucket: None)

    client.historical_data(
        "12345",
        "2026-06-29 09:15:00",
        "2026-06-29 09:16:00",
        "minute",
    )

    assert captured["path"] == "/instruments/historical/12345/minute"
    client._client.close()


@pytest.mark.parametrize(
    "bad_token",
    ["NFO:NIFTY26JUNFUT", "NSE:NIFTY", "None", "", None, 0, -1, "12x"],
)
def test_historical_data_rejects_invalid_token_before_http(
    monkeypatch: pytest.MonkeyPatch,
    bad_token: object,
) -> None:
    client = _make_client()
    calls = []
    monkeypatch.setattr(client, "_make_request", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(
        client, "_acquire_bucket", lambda _bucket: calls.append("bucket")
    )

    with pytest.raises(BrokerError, match="positive integer"):
        client.historical_data(  # type: ignore[arg-type]
            bad_token,
            "2026-06-29 09:15:00",
            "2026-06-29 09:16:00",
            "minute",
        )

    assert calls == []
    client._client.close()


def test_format_kite_datetime_timezone_cases() -> None:
    assert (
        _format_kite_datetime(datetime(2026, 6, 29, 6, 25, tzinfo=timezone.utc))
        == "2026-06-29 11:55:00"
    )
    assert (
        _format_kite_datetime(
            datetime(
                2026,
                6,
                29,
                11,
                55,
                tzinfo=timezone(timedelta(hours=5, minutes=30)),
            )
        )
        == "2026-06-29 11:55:00"
    )
    assert (
        _format_kite_datetime(
            datetime(2026, 6, 29, 2, 25, tzinfo=timezone(timedelta(hours=-4)))
        )
        == "2026-06-29 11:55:00"
    )
    assert _format_kite_datetime(datetime(2026, 6, 29, 11, 55)) == "2026-06-29 11:55:00"
    assert _format_kite_datetime("2026-06-29 11:55:00") == "2026-06-29 11:55:00"
    with pytest.raises(BrokerError):
        _format_kite_datetime("   ")


@pytest.mark.parametrize(
    ("status", "expect_order", "exc_type", "text"),
    [
        (400, False, BrokerError, "bad input"),
        (400, True, OrderPlacementError, "bad input"),
        (404, True, OrderPlacementError, "bad input"),
        (401, False, ConfigurationError, "authentication failed"),
        (403, False, ConfigurationError, "access denied"),
        (429, False, BrokerError, "rate limit"),
        (500, False, BrokerError, "server exploded"),
    ],
)
def test_raise_for_status_maps_errors_without_unboundlocal(
    status: int,
    expect_order: bool,
    exc_type: type[Exception],
    text: str,
) -> None:
    client = _make_client()
    response = httpx.Response(
        status,
        json={"message": "bad input" if status != 500 else "server exploded"},
        request=httpx.Request("GET", "https://kite.test/x"),
    )

    with pytest.raises(exc_type) as err:
        client._raise_for_status(response, expect_order)

    assert text in str(err.value).lower()
    client._client.close()


def test_authentication_classifier_requires_strong_evidence() -> None:
    assert ZerodhaKiteClient._is_authentication_failure(
        status_code=None, payload={"error_type": "TokenException"}, error_text=""
    )
    assert ZerodhaKiteClient._is_authentication_failure(
        status_code=None, payload={"message": "invalid session"}, error_text=""
    )
    assert ZerodhaKiteClient._is_authentication_failure(
        status_code=None, payload={"message": "incorrect access_token"}, error_text=""
    )
    assert not ZerodhaKiteClient._is_authentication_failure(
        status_code=None,
        payload={"error_type": "InputException", "message": "invalid instrument token"},
        error_text="",
    )
    assert not ZerodhaKiteClient._is_authentication_failure(
        status_code=None, payload={"message": "invalid token"}, error_text=""
    )

from __future__ import annotations

import logging
import math

import httpx
import pytest

from nifty_scalper_bot.data.rest.zerodha_client import ZerodhaKiteClient
from nifty_scalper_bot.utils.errors import (
    BrokerAuthenticationError,
    BrokerBalanceUnavailableError,
    ConfigurationError,
)


def test_balance_refresh_logs_amount_fields(caplog) -> None:
    client = ZerodhaKiteClient(api_key='k', access_token='t')
    client.get_account_margins = lambda segment='equity': {
        'equity': {
            'net': 125000.0,
            'available': {
                'cash': 100000.0,
                'live_balance': 98000.0,
                'opening_balance': 102000.0,
            },
        }
    }
    with caplog.at_level(logging.INFO):
        out = client.get_available_balance('equity')

    assert out == 100000.0
    assert any(
        'ZERODHA_BALANCE_REFRESH_SUCCESS' in r.getMessage()
        for r in caplog.records
    )
    assert any(
        getattr(r, 'event', '') == 'ZERODHA_BALANCE_REFRESH_SUCCESS'
        for r in caplog.records
    )


def test_balance_refresh_unchanged_is_debug(monkeypatch, caplog) -> None:
    client = ZerodhaKiteClient(api_key='k', access_token='t')
    payload = {
        'equity': {
            'net': 50000.0,
            'available': {
                'cash': 45000.0,
                'live_balance': 44000.0,
                'opening_balance': 46000.0,
            },
        }
    }
    client.get_account_margins = lambda segment='equity': payload

    with caplog.at_level(logging.INFO):
        client.get_available_balance('equity')
    caplog.clear()

    # Keep time inside 15m unchanged-info window.
    monkeypatch.setattr(client, '_log_time_fn', lambda: 1.0)
    with caplog.at_level(logging.DEBUG):
        client.get_available_balance('equity')

    assert any(
        'ZERODHA_BALANCE_REFRESH_UNCHANGED' in r.getMessage()
        for r in caplog.records
    )



def test_http_403_latches_and_second_balance_call_skips_http(monkeypatch) -> None:
    client = ZerodhaKiteClient(api_key="k", access_token="t")
    calls = {"count": 0}

    def fake_request(method, url, **kwargs):  # noqa: ANN001
        calls["count"] += 1
        return httpx.Response(
            403,
            json={
                "status": "error",
                "message": "Incorrect api_key or access_token",
                "error_type": "TokenException",
            },
            request=httpx.Request(method, "https://api.kite.trade" + url),
        )

    monkeypatch.setattr(client._client, "request", fake_request)
    with pytest.raises(BrokerAuthenticationError):
        client.get_available_balance("equity")
    assert client.auth_invalid is True
    assert calls["count"] == 1
    with pytest.raises(BrokerAuthenticationError):
        client.get_available_balance("equity")
    assert calls["count"] == 1
    client._client.close()


def test_available_balance_never_calls_legacy_margins_endpoint(monkeypatch) -> None:
    client = ZerodhaKiteClient(api_key="k", access_token="t")
    seen: list[str] = []

    def fake_make(method, endpoint, **kwargs):  # noqa: ANN001
        seen.append(endpoint)
        return {
            "data": {
                "equity": {
                    "net": 0.0,
                    "available": {
                        "cash": 0.0,
                        "live_balance": 0.0,
                        "opening_balance": 0.0,
                    },
                    "utilised": {},
                }
            }
        }

    monkeypatch.setattr(client, "_make_request", fake_make)
    assert client.get_available_balance("equity") == 0.0
    assert "/user/margins/equity" in seen
    assert "/margins/equity" not in seen
    client._client.close()


def test_invalid_balance_payload_and_non_finite_raise_typed_errors() -> None:
    client = ZerodhaKiteClient(api_key="k", access_token="t")
    with pytest.raises(BrokerBalanceUnavailableError):
        client._parse_account_margin_summary(
            {"equity": {"available": {}}}, segment="equity"
        )
    with pytest.raises(BrokerBalanceUnavailableError):
        client._parse_account_margin_summary(
            {"equity": {"available": {"cash": math.inf}, "utilised": {}}},
            segment="equity",
        )
    client._client.close()


def test_no_live_env_balance_fallback(monkeypatch) -> None:
    monkeypatch.setenv("RISK_CAPITAL", "1000000")
    client = ZerodhaKiteClient(api_key="k", access_token="t")
    def fail_account_margins(segment="equity"):  # noqa: ANN001
        raise RuntimeError("broker down")

    client.get_account_margins = fail_account_margins
    with pytest.raises(BrokerBalanceUnavailableError):
        client.get_available_balance("equity")
    client._client.close()


def test_simulated_balance_requires_explicit_capital(monkeypatch) -> None:
    for key in ("RISK__CAPITAL", "RISK_CAPITAL", "BACKTEST__CAPITAL"):
        monkeypatch.delenv(key, raising=False)
    client = ZerodhaKiteClient(api_key="k", access_token="t")
    with pytest.raises(ConfigurationError):
        client._resolve_simulated_balance()
    monkeypatch.setenv("BACKTEST__CAPITAL", "12345")
    assert client._resolve_simulated_balance() == 12345.0
    client._client.close()

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


def test_auth_latch_reprobes_and_self_heals_on_success() -> None:
    """2026-07-07 incident: IP-allowlist auth failure latched _auth_invalid
    forever (only cleared in __init__), so even after fixing the Kite console
    the bot stayed blocked until restart, while the quote poller logged one
    ERROR per second. The latch must let one re-probe through per interval
    and clear itself on the first successful authenticated request."""
    import threading

    from nifty_scalper_bot.data.rest.zerodha_client import (
        BrokerAuthenticationError,
        ZerodhaKiteClient,
    )

    client = ZerodhaKiteClient.__new__(ZerodhaKiteClient)
    client._auth_invalid = True
    client._auth_invalid_reason = "ip blocked"
    client._auth_invalid_at = 1.0
    client._auth_reprobe_interval = 60.0
    client._auth_reprobe_next = 0.0
    clock = {"now": 1000.0}
    client._log_time_fn = lambda: clock["now"]

    client._raise_if_authentication_latched()  # re-probe allowed
    with pytest.raises(BrokerAuthenticationError):
        client._raise_if_authentication_latched()  # inside window: blocked
    clock["now"] = 1061.0
    client._raise_if_authentication_latched()  # next window: re-probe again

    client._resilience_lock = threading.Lock()
    client._transient_error_streak = 3
    client._breaker_open_until = 9.0
    client._auth_failure_generation = 1
    client._reset_transient_state()  # success path
    assert client._auth_invalid is False
    client._raise_if_authentication_latched()  # gate open again


def test_settings_import_survives_malformed_env(monkeypatch) -> None:
    """2026-07-09 incident: a malformed tuning env var killed the settings
    module mid-import; the partial module then surfaced as cryptic
    AttributeErrors (USE_REGIME_ADAPTIVE, REGIME_TREND_SIZING_MULT) and the
    bot entered degraded mode. Tunables must fail soft to their defaults."""
    import importlib

    monkeypatch.setenv("REGIME_TREND_SIZING_MULT", "not-a-number")
    monkeypatch.setenv("MAX_LOTS_PER_TRADE", "banana")
    monkeypatch.setenv("MIN_LOTS_PER_TRADE", "0")  # below minimum -> clamp
    import nifty_scalper_bot.config.settings as settings_module

    reloaded = importlib.reload(settings_module)
    try:
        assert reloaded.REGIME_TREND_SIZING_MULT == 1.15
        assert reloaded.MAX_LOTS_PER_TRADE == 2
        assert reloaded.MIN_LOTS_PER_TRADE == 1
        assert reloaded.USE_REGIME_ADAPTIVE is True
    finally:
        monkeypatch.delenv("REGIME_TREND_SIZING_MULT")
        monkeypatch.delenv("MAX_LOTS_PER_TRADE")
        monkeypatch.delenv("MIN_LOTS_PER_TRADE")
        importlib.reload(settings_module)


def test_spot_index_token_resolves_via_zerodha_full_name() -> None:
    """NSE:NIFTY must resolve although Zerodha lists the index as 'NIFTY 50'."""
    import logging as _logging

    from nifty_scalper_bot.data.market_data_manager import MarketDataManager

    mdm = MarketDataManager.__new__(MarketDataManager)
    mdm._logger = _logging.getLogger("test")
    mdm._token_by_symbol = {}
    mdm._canonical_symbol = lambda s: s
    registered: dict = {}
    mdm.register_symbol = lambda s, t: registered.__setitem__(s, t)

    class _Broker:
        def instruments(self, exchange):
            assert exchange == "NSE"
            return [{"tradingsymbol": "NIFTY 50", "instrument_token": 256265}]

    mdm._broker = _Broker()
    assert mdm._resolve_token("NSE:NIFTY") == 256265
    assert registered["NSE:NIFTY"] == 256265

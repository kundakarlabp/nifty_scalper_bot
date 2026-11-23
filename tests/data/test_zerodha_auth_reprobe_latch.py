from __future__ import annotations

import logging

import httpx
import pytest

from nifty_scalper_bot.data.rest.zerodha_client import ZerodhaKiteClient
from nifty_scalper_bot.utils.errors import BrokerAuthenticationError


def test_failed_auth_reprobe_keeps_latch_and_never_logs_restored(monkeypatch, caplog) -> None:
    """A second 403 is not authentication recovery.

    Regression for the 2026-08-18 startup loop where a failed reprobe called
    ``_reset_transient_state`` before classifying the same response, emitted
    ZERODHA_AUTH_RESTORED, then immediately invalidated auth again.
    """
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
    assert client.authentication_status_snapshot()["generation"] == 1
    assert calls["count"] == 1

    # Force the bounded reprobe window open and exercise the request boundary
    # directly. Higher-level balance helpers intentionally have additional
    # fail-fast guards; the defect itself is response classification inside
    # _make_request after a permitted reprobe reaches Zerodha.
    client._auth_reprobe_next = 0.0
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        with pytest.raises(BrokerAuthenticationError):
            client._make_request(
                "GET",
                "/user/margins/equity",
                operation_label="auth.reprobe.regression",
            )

    assert calls["count"] == 2
    assert client.auth_invalid is True
    assert client.authentication_status_snapshot()["generation"] == 1
    assert not any(
        getattr(record, "event", "") == "ZERODHA_AUTH_RESTORED"
        or "ZERODHA_AUTH_RESTORED" in record.getMessage()
        for record in caplog.records
    )
    client._client.close()


def test_public_instrument_success_cannot_restore_auth_latch(
    monkeypatch, caplog
) -> None:
    """Unauthenticated instrument dumps are not proof that a Kite token recovered."""
    client = ZerodhaKiteClient(api_key="k", access_token="t")
    calls: list[str] = []

    def fake_request(method, url, **kwargs):  # noqa: ANN001
        calls.append(url)
        if url == "/user/margins/equity":
            return httpx.Response(
                403,
                json={
                    "status": "error",
                    "message": "Incorrect api_key or access_token",
                    "error_type": "TokenException",
                },
                request=httpx.Request(method, "https://api.kite.trade" + url),
            )
        if url == "/instruments/NFO":
            return httpx.Response(
                200,
                text=(
                    "instrument_token,exchange_token,tradingsymbol,name,"
                    "last_price,expiry,strike,tick_size,lot_size,"
                    "instrument_type,segment,exchange\n"
                ),
                request=httpx.Request(method, "https://api.kite.trade" + url),
            )
        if url == "/user/profile":
            return httpx.Response(
                200,
                json={"status": "success", "data": {"user_id": "AB1234"}},
                request=httpx.Request(method, "https://api.kite.trade" + url),
            )
        raise AssertionError(f"unexpected endpoint: {url}")

    monkeypatch.setattr(client._client, "request", fake_request)

    with pytest.raises(BrokerAuthenticationError):
        client.get_available_balance("equity")
    assert client.auth_invalid is True

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        instrument_response = client._make_request(
            "GET",
            "/instruments/NFO",
            raw_response=True,
            operation_label="auth.public_instrument.regression",
        )

    assert instrument_response.status_code == 200
    assert client.auth_invalid is True
    assert not any(
        getattr(record, "event", "") == "ZERODHA_AUTH_RESTORED"
        or "ZERODHA_AUTH_RESTORED" in record.getMessage()
        for record in caplog.records
    )

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert client.get_profile()["user_id"] == "AB1234"

    assert client.auth_invalid is False
    assert calls == [
        "/user/margins/equity",
        "/instruments/NFO",
        "/user/profile",
    ]
    assert any(
        getattr(record, "event", "") == "ZERODHA_AUTH_RESTORED"
        or "ZERODHA_AUTH_RESTORED" in record.getMessage()
        for record in caplog.records
    )
    client._client.close()

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

    # Force the bounded reprobe window open. That reprobe also fails
    # authentication, so the original invalid generation must remain latched
    # and no false RESTORED event may be emitted.
    client._auth_reprobe_next = 0.0
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        with pytest.raises(BrokerAuthenticationError):
            client.get_available_balance("equity")

    assert calls["count"] == 2
    assert client.auth_invalid is True
    assert client.authentication_status_snapshot()["generation"] == 1
    assert not any(
        getattr(record, "event", "") == "ZERODHA_AUTH_RESTORED"
        or "ZERODHA_AUTH_RESTORED" in record.getMessage()
        for record in caplog.records
    )
    client._client.close()

from __future__ import annotations

import logging

from nifty_scalper_bot.data.rest.zerodha_client import ZerodhaKiteClient


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
    assert any('ZERODHA_BALANCE_REFRESH_SUCCESS' in r.getMessage() for r in caplog.records)
    assert any(getattr(r, 'event', '') == 'ZERODHA_BALANCE_REFRESH_SUCCESS' for r in caplog.records)


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

    assert any('ZERODHA_BALANCE_REFRESH_UNCHANGED' in r.getMessage() for r in caplog.records)

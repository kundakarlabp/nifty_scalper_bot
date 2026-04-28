from __future__ import annotations

from nifty_scalper_bot.data.market_data_policy import MarketDataPolicy


def test_market_data_policy_defaults(monkeypatch) -> None:
    monkeypatch.delenv('MARKETDATA_PRIMARY_SOURCE', raising=False)
    monkeypatch.delenv('NIFTY_INTERNAL_SYMBOL', raising=False)
    monkeypatch.delenv('NIFTY_ZERODHA_QUOTE_SYMBOL', raising=False)
    monkeypatch.delenv('MARKETDATA_REST_FALLBACK_MODE', raising=False)
    policy = MarketDataPolicy.from_env()
    assert policy.primary_source == 'websocket'
    assert policy.nifty_internal_symbol == 'NSE:NIFTY'
    assert policy.nifty_zerodha_quote_symbol == 'NSE:NIFTY 50'
    assert policy.rest_fallback_mode == 'after_ws_degraded'


def test_policy_quote_alias_and_suppression(monkeypatch) -> None:
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    monkeypatch.setenv('ENABLE_LIVE', 'true')
    policy = MarketDataPolicy.from_env()
    assert policy.to_broker_quote_symbol('NSE:NIFTY') == 'NSE:NIFTY 50'
    assert policy.to_broker_quote_symbol('NIFTY') == 'NSE:NIFTY 50'
    assert 'NSE:NIFTY 50' in policy.quote_aliases('NSE:NIFTY')
    assert (
        policy.should_suppress_rest_quote(
            symbol='NSE:NIFTY', ws_started=False, fallback_active=False
        )
        is True
    )
    assert (
        policy.should_suppress_rest_quote(
            symbol='NSE:NIFTY', ws_started=True, fallback_active=True
        )
        is False
    )

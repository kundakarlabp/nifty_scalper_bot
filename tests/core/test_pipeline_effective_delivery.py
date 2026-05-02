from __future__ import annotations

import logging
from types import SimpleNamespace

from nifty_scalper_bot.core.app import _emit_option_symbol_pipeline_status


class StubHub:
    _mdm_subscribed_symbols: set[str] = set()

    def __init__(self) -> None:
        self._tick_subscribers = {}
        self._tick_subscribers_by_token = {123: {object()}}

    def _canonical_quote_symbol(self, symbol: str) -> str:
        return symbol

    def debug_subscription_status(self, symbol: str, token: int | None = None):
        return {
            'symbol': symbol,
            'canonical': symbol,
            'token': token,
            'symbol_callbacks': 0,
            'token_callbacks': 1,
            'aliases': [],
            'quote_present': False,
            'token_quote_present': True,
        }


def test_effective_delivery_uses_token_flags(caplog) -> None:
    ctx = SimpleNamespace(
        data_hub=StubHub(),
        market_data_manager=SimpleNamespace(
            _tracked_symbols=set(), _subscribers={}, _active_subscribed_symbols=set(),
            _last_tick_time={}, _subscribed_tokens=set(), _requested_tokens=set(), _active_tokens=set(), _desired_tokens={123}
        ),
        strategy_runner=None,
        datahub_runner_subscriptions={'TOKEN:123'},
        message_bus_tick_owner='data_hub',
    )
    caplog.set_level(logging.INFO)
    _emit_option_symbol_pipeline_status(ctx, symbol='NFO:X', token=123, selected=True, hydrated_bars=0, runner_added=False, source='test', reason='test')
    rec = next(r for r in caplog.records if r.message.startswith('OPTION_SYMBOL_PIPELINE_STATUS'))
    assert rec.effective_datahub_delivery is True
    assert rec.effective_quote_present is True

from __future__ import annotations

import logging
from types import SimpleNamespace

from nifty_scalper_bot.core.app import _emit_option_symbol_pipeline_status


class StubDataHub:
    def __init__(self) -> None:
        self._tick_subscribers: dict[str, set[object]] = {}
        self._tick_subscribers_by_token: dict[int, set[object]] = {123: {object()}}
        self._mdm_subscribed_symbols: set[str] = set()

    def _canonical_quote_symbol(self, symbol: str) -> str:
        return symbol

    def get_quote(self, symbol: str, allow_pull: bool = False):
        return None

    def get_tick_by_token(self, token: int):
        return {'instrument_token': token}


class StubMdm:
    _tracked_symbols: set[str] = set()
    _subscribers: dict[str, set[object]] = {}
    _active_subscribed_symbols: set[str] = set()
    _last_tick_time: dict[str, float] = {}
    _subscribed_tokens: set[int] = set()
    _requested_tokens: set[int] = set()
    _active_tokens: set[int] = set()
    _desired_tokens: set[int] = {123}


def test_pipeline_status_uses_token_callbacks(caplog) -> None:
    ctx = SimpleNamespace(
        data_hub=StubDataHub(),
        market_data_manager=StubMdm(),
        strategy_runner=None,
        datahub_runner_subscriptions={'TOKEN:123'},
        message_bus_tick_owner='data_hub',
    )
    caplog.set_level(logging.INFO)

    _emit_option_symbol_pipeline_status(
        ctx,
        symbol='NFO:NIFTY26MAY23950CE',
        token=123,
        selected=True,
        hydrated_bars=0,
        runner_added=False,
        source='test',
        reason='test',
    )

    rec = next(r for r in caplog.records if r.message.startswith('OPTION_SYMBOL_PIPELINE_STATUS'))
    assert rec.datahub_callback_registered is True
    assert rec.datahub_quote_present is True
    assert rec.datahub_runner_subscription_registered is True

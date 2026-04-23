from __future__ import annotations

import inspect
from pathlib import Path

from nifty_scalper_bot.core import app as core_app


def test_app_has_no_direct_websocket_subscription_calls() -> None:
    source = inspect.getsource(core_app)

    assert 'ctx.websocket_manager.subscribe_tokens(' not in source
    assert 'ctx.websocket_manager.unsubscribe_tokens(' not in source
    assert 'ws.subscribe_tokens(' not in source
    assert 'ctx.streamer.subscribe([tok])' not in source
    assert 'ctx.streamer.unsubscribe([tok])' not in source


def test_app_routes_startup_and_universe_subscriptions_via_mdm() -> None:
    source = inspect.getsource(core_app)

    assert 'mdm.request_token_subscriptions(tokens_to_poll)' in source
    assert 'ctx.market_data_manager.request_token_subscription(' in source
    assert 'ctx.market_data_manager.request_token_unsubscription(' in source


def test_startup_tracking_is_deferred_without_immediate_subscription() -> None:
    source = inspect.getsource(core_app)

    assert 'subscribe=False' in source
    assert 'mdm.ensure_tracking(' in source


def test_startup_listener_subscriptions_flush_after_universe_prep() -> None:
    source = inspect.getsource(core_app)

    assert 'data_hub.subscribe_ticks(regime_symbol, callback)' in source
    assert 'data_hub.subscribe_ticks(risk_symbol, _risk_state_tick_listener)' in source
    assert 'ctx.data_hub.flush_pending_live_subscriptions()' in source

    flush_idx = source.index('ctx.data_hub.flush_pending_live_subscriptions()')
    token_idx = source.index('mdm.request_token_subscriptions(tokens_to_poll)')
    assert flush_idx < token_idx


def test_market_data_manager_supports_optional_subscription_during_tracking() -> None:
    source = Path(
        'src/nifty_scalper_bot/data/market_data_manager.py'
    ).read_text(encoding='utf-8')

    assert 'def ensure_tracking(' in source
    assert 'subscribe: bool = True' in source

from __future__ import annotations

import inspect

from nifty_scalper_bot.core import app as core_app


def test_app_has_no_direct_websocket_subscription_calls() -> None:
    source = inspect.getsource(core_app)

    assert 'ctx.websocket_manager.subscribe_tokens(' not in source
    assert 'ctx.websocket_manager.unsubscribe_tokens(' not in source
    assert 'ws.subscribe_tokens(' not in source


def test_app_routes_startup_and_universe_subscriptions_via_mdm() -> None:
    source = inspect.getsource(core_app)

    assert 'mdm.request_token_subscriptions(tokens_to_poll)' in source
    assert 'ctx.market_data_manager.request_token_subscription(' in source

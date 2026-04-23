from __future__ import annotations

import inspect

from nifty_scalper_bot.core import app as core_app


def test_no_early_live_subscription_for_regime_and_risk_listeners() -> None:
    source = inspect.getsource(core_app)

    regime_idx = source.index('data_hub.subscribe_ticks(regime_symbol, callback)')
    risk_idx = source.index(
        'data_hub.subscribe_ticks(risk_symbol, _risk_state_tick_listener)'
    )
    flush_idx = source.index('ctx.data_hub.flush_pending_live_subscriptions()')
    request_idx = source.index('mdm.request_token_subscriptions(tokens_to_poll)')

    assert regime_idx < flush_idx
    assert risk_idx < flush_idx
    assert flush_idx < request_idx


def test_flush_happens_after_universe_building_milestones() -> None:
    source = inspect.getsource(core_app)

    options_store_idx = source.index(
        'ctx.options_store = OptionsContractStore(ctx.instrument_manager)'
    )
    basket_idx = source.index('_build_canonical_active_basket(')
    flush_idx = source.index('ctx.data_hub.flush_pending_live_subscriptions()')

    assert options_store_idx < flush_idx
    assert basket_idx < flush_idx

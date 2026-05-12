from __future__ import annotations

from types import SimpleNamespace

import pytest

from nifty_scalper_bot.core import app


class _Runner:
    def __init__(self) -> None:
        self.calls = []
        self._running = True
        self._indicator_engine = SimpleNamespace(get_history=lambda s: list(range(60)))

    def set_runtime_readiness(self, **kwargs):
        self.calls.append(kwargs)

    def set_active_trading_universe(self, basket):
        self.basket = basket


def _ctx(mdm):
    return SimpleNamespace(
        settings=SimpleNamespace(execution_mode='LIVE'),
        market_data_manager=mdm,
        strategy_runner=_Runner(),
        broker_client=object(),
        order_manager=object(),
        active_trading_universe={},
    )


def test_selected_options_derived_from_basket() -> None:
    ce, pe = app._pick_atm_option_symbols_from_basket({
        'option_symbols': ['NFO:NIFTY24500CE', 'NFO:NIFTY24600PE', 'NFO:NIFTY24600CE'],
        'atm_strike': 24600,
    })
    assert ce == 'NFO:NIFTY24600CE'
    assert pe == 'NFO:NIFTY24600PE'


def test_pick_atm_option_symbols_fallbacks_to_symbols() -> None:
    ce, pe = app._pick_atm_option_symbols_from_basket({
        'symbols': ['NSE:NIFTY', 'NFO:NIFTY24600CE', 'NFO:NIFTY24600PE'],
        'atm_strike': 24600,
    })
    assert ce == 'NFO:NIFTY24600CE'
    assert pe == 'NFO:NIFTY24600PE'


@pytest.mark.asyncio
async def test_recompute_readiness_arms_with_spot_selected_ce_pe() -> None:
    class _Snap:
        def __init__(self) -> None:
            self.ltp = 1.0
            self.tick_age_s = 1.0

    mdm = SimpleNamespace(
        get_ohlc_bars=lambda s: list(range(40)) if s != 'NSE:NIFTY' else list(range(40)),
        get_symbol_snapshot=lambda s: _Snap() if s in {'NSE:NIFTY', 'NFO:NIFTY24600CE', 'NFO:NIFTY24600PE'} else None,
    )
    ctx = _ctx(mdm)
    ctx.active_trading_universe = {
        'spot_symbol': 'NSE:NIFTY',
        'atm_strike': 24600,
        'option_symbols': ['NFO:NIFTY24600CE', 'NFO:NIFTY24600PE', 'NFO:NIFTY24700CE'],
    }
    await app._recompute_and_push_runtime_readiness(ctx, reason='test')
    assert ctx.data_hard_ready is True


@pytest.mark.asyncio
async def test_runtime_readiness_arms_with_option_candles() -> None:
    mdm = SimpleNamespace(
        get_ohlc_bars=lambda s: [1, 2, 3] if s.startswith('NFO:') else [1],
        get_symbol_snapshot=lambda s: None,
    )
    ctx = _ctx(mdm)
    ctx.active_trading_universe = {
        'spot_symbol': 'NSE:NIFTY',
        'selected_ce': 'NFO:NIFTY24600CE',
        'selected_pe': 'NFO:NIFTY24600PE',
        'option_symbols': ['NFO:NIFTY24600CE', 'NFO:NIFTY24600PE'],
    }
    await app._recompute_and_push_runtime_readiness(ctx, reason='test')
    assert ctx.live_orders_armed is False


def test_underhydrated_symbols_not_added_to_runner() -> None:
    runner = SimpleNamespace(_required_candles=20, add_symbol=lambda s: (_ for _ in ()).throw(AssertionError('must not add')))
    mdm = SimpleNamespace(_min_required_bars=20, get_ohlc_bars=lambda s: [], get_symbol_snapshot=lambda s: None)
    ctx = SimpleNamespace(strategy_runner=runner, market_data_manager=mdm, data_hub=None, datahub_runner_subscriptions=set())
    pending = set()
    assert app._gate_runner_symbol_add(ctx, 'NFO:NIFTY24600CE', pending) is False


def test_gate_runner_symbol_add_discards_pending_when_quote_ready() -> None:
    runner = SimpleNamespace(_required_candles=20, added=[], add_symbol=lambda s: runner.added.append(s))
    mdm = SimpleNamespace(get_ohlc_bars=lambda s: [], get_symbol_snapshot=lambda s: {'ltp': 1})
    ctx = SimpleNamespace(strategy_runner=runner, market_data_manager=mdm, data_hub=None, datahub_runner_subscriptions=set())
    pending = {'NFO:NIFTY24600CE'}
    assert app._gate_runner_symbol_add(ctx, 'NFO:NIFTY24600CE', pending) is True
    assert 'NFO:NIFTY24600CE' not in pending


@pytest.mark.asyncio
async def test_dynamic_basket_commit_sets_selected_symbols_and_readiness() -> None:
    class _Snap:
        def __init__(self) -> None:
            self.ltp = 1.0
            self.tick_age_s = 1.0
            self.tradable_quote = True
            self.bid = 0.9
            self.ask = 1.1

    mdm = SimpleNamespace(
        get_ohlc_bars=lambda s: list(range(60)),
        get_symbol_snapshot=lambda s: _Snap(),
    )
    ctx = _ctx(mdm)
    ctx.active_trading_universe = {'spot_symbol': 'NSE:NIFTY', 'atm_strike': 24600}
    app._commit_active_dynamic_basket(
        ctx,
        basket=ctx.active_trading_universe,
        option_symbols=['NFO:NIFTY24600CE', 'NFO:NIFTY24600PE'],
        symbols=['NSE:NIFTY', 'NFO:NIFTY24600CE', 'NFO:NIFTY24600PE'],
        atm_strike=24600,
    )
    await app._recompute_and_push_runtime_readiness(ctx, reason='dynamic_basket_committed')
    assert ctx.selected_ce == 'NFO:NIFTY24600CE'
    assert ctx.selected_pe == 'NFO:NIFTY24600PE'
    assert ctx.data_hard_ready is True
    assert ctx.evaluation_ready is True

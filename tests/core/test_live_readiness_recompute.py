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

    def sync_history_from_mdm(self, symbol, **kwargs):
        bars = len(self._indicator_engine.get_history(symbol) or [])
        return SimpleNamespace(success=True, runner_bars=bars, indicator_bars=bars)


def _ctx(mdm):
    return SimpleNamespace(
        settings=SimpleNamespace(execution_mode='LIVE'),
        market_data_manager=mdm,
        strategy_runner=_Runner(),
        broker_client=SimpleNamespace(is_connected=lambda: True),
        order_manager=object(),
        data_hub=SimpleNamespace(
            get_subscription_snapshot=lambda symbol: SimpleNamespace(
                state=SimpleNamespace(value="live"), generation=1, updated_at=1.0
            )
        ),
        active_trading_universe={},
    )


async def _ok_history(*args, **kwargs):
    return SimpleNamespace(failure_reason=None, fetched_rows=0, accepted_rows=0)


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
        hydrate_active_contract_basket=lambda basket=None: {"hard_ready": True, "missing": [], "symbols": {}},
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
        hydrate_active_contract_basket=lambda basket=None: {"hard_ready": True, "missing": [], "symbols": {}},
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
        hydrate_active_contract_basket=lambda basket=None: {"hard_ready": True, "missing": [], "symbols": {}},
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


@pytest.mark.asyncio
async def test_selected_option_ltp_only_blocks_live_execution_bid_ask_missing(monkeypatch) -> None:
    monkeypatch.setattr(app, 'get_market_state', lambda: app.MarketState.OPEN)

    class _Snap:
        ltp = 100.0
        tick_age_s = 1.0
        bid = None
        ask = None
        tradable_quote = False
        depth_available = False

    mdm = SimpleNamespace(
        get_ohlc_bars=lambda s, limit=None: list(range(40)),
        get_symbol_snapshot=lambda s: _Snap(),
        hydrate_active_contract_basket=lambda basket=None: {"hard_ready": True, "missing": [], "symbols": {}},
    )
    ctx = _ctx(mdm)
    ctx.active_trading_universe = {
        'spot_symbol': 'NSE:NIFTY',
        'selected_ce': 'NFO:NIFTY24600CE',
        'selected_pe': 'NFO:NIFTY24600PE',
        'option_symbols': ['NFO:NIFTY24600CE', 'NFO:NIFTY24600PE'],
    }

    await app._recompute_and_push_runtime_readiness(ctx, reason='test')

    assert ctx.evaluation_ready is True
    assert ctx.live_orders_armed is False
    assert ctx.execution_ready_by_symbol == {
        'NFO:NIFTY24600CE': False,
        'NFO:NIFTY24600PE': False,
    }
    assert ctx.live_block_reason == 'execution_not_armed:selected_option_quote_missing'


@pytest.mark.asyncio
async def test_selected_option_history_cold_clears_when_both_selected_histories_ready(monkeypatch, caplog) -> None:
    monkeypatch.setattr(app, 'get_market_state', lambda: app.MarketState.OPEN)

    class _Snap:
        ltp = 100.0
        tick_age_s = 1.0
        bid = 99.0
        ask = 101.0
        tradable_quote = True
        depth_available = True

    mdm = SimpleNamespace(
        get_ohlc_bars=lambda s, limit=None: list(range(118)),
        get_symbol_snapshot=lambda s: _Snap(),
        hydrate_active_contract_basket=lambda basket=None: {
            "hard_ready": False,
            "missing": ["selected_option_history_cold"],
            "symbols": {},
        },
        ensure_history=_ok_history,
    )
    ctx = _ctx(mdm)
    ctx.active_symbol_tokens = {'NFO:NIFTY24600CE': 1, 'NFO:NIFTY24600PE': 2}
    ctx.active_trading_universe = {
        'spot_symbol': 'NSE:NIFTY',
        'selected_ce': 'NFO:NIFTY24600CE',
        'selected_pe': 'NFO:NIFTY24600PE',
        'option_symbols': ['NFO:NIFTY24600CE', 'NFO:NIFTY24600PE'],
    }

    with caplog.at_level('INFO'):
        await app._recompute_and_push_runtime_readiness(ctx, reason='test')

    assert 'selected_option_history_cold' not in (ctx.live_block_reason or '')
    assert any('READINESS_BLOCKER_CLEARED blocker=selected_option_history_cold' in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_live_orders_armed_becomes_true_after_history_ready(monkeypatch) -> None:
    monkeypatch.setattr(app, 'get_market_state', lambda: app.MarketState.OPEN)

    class _Snap:
        ltp = 100.0
        tick_age_s = 1.0
        bid = 99.0
        ask = 101.0
        tradable_quote = True
        depth_available = True

    mdm = SimpleNamespace(
        get_ohlc_bars=lambda s, limit=None: list(range(118)),
        get_symbol_snapshot=lambda s: _Snap(),
        hydrate_active_contract_basket=lambda basket=None: {"hard_ready": True, "missing": [], "symbols": {}},
        ensure_history=_ok_history,
    )
    ctx = _ctx(mdm)
    ctx.active_symbol_tokens = {'NFO:NIFTY24600CE': 1, 'NFO:NIFTY24600PE': 2}
    ctx.active_trading_universe = {
        'spot_symbol': 'NSE:NIFTY',
        'selected_ce': 'NFO:NIFTY24600CE',
        'selected_pe': 'NFO:NIFTY24600PE',
        'option_symbols': ['NFO:NIFTY24600CE', 'NFO:NIFTY24600PE'],
    }

    await app._recompute_and_push_runtime_readiness(ctx, reason='test')

    assert ctx.live_orders_armed is True
    assert ctx.live_block_reason is None
    assert ctx.strategy_runner.calls[-1]['live_orders_armed'] is True

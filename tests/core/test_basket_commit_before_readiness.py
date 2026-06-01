from types import SimpleNamespace

import pytest

from nifty_scalper_bot.core import app


class _IM:
    def is_loaded(self): return True


class _Broker:
    def get_instrument_token(self, symbol: str) -> int: return 123


class _MDM:
    def get_ohlc_bars(self, symbol, limit=None): return [1] * 30
    def get_symbol_snapshot(self, symbol): return SimpleNamespace(ltp=100.0, bid=99.0, ask=101.0, tradable_quote=True, tick_age_s=0.1)
    async def hydrate_symbol_history(self, *args, **kwargs): return []
    def get_active_nifty_future_symbol_cached(self): return "NFO:NIFTY26JUNFUT"


@pytest.mark.asyncio
async def test_build_commits_selected_before_readiness(monkeypatch):
    basket = {
        'selected_ce': 'NFO:NIFTY26MAY23700CE',
        'selected_pe': 'NFO:NIFTY26MAY23700PE',
        'atm_ce': 'NFO:NIFTY26MAY23700CE',
        'atm_pe': 'NFO:NIFTY26MAY23700PE',
        'atm_strike': 23700,
        'symbols': ['NSE:NIFTY', 'NFO:NIFTY26MAY23700CE', 'NFO:NIFTY26MAY23700PE'],
        'option_symbols': ['NFO:NIFTY26MAY23700CE', 'NFO:NIFTY26MAY23700PE'],
    }
    monkeypatch.setattr(app, '_build_canonical_active_basket', lambda **kwargs: basket)
    monkeypatch.setattr(app, 'get_market_state', lambda: app.MarketState.CLOSED)
    ctx = SimpleNamespace(
        instrument_manager=_IM(),
        market_data_manager=_MDM(),
        broker_client=_Broker(),
        settings=SimpleNamespace(option_universe=SimpleNamespace(strike_step=50), execution_mode='LIVE'),
        strategy_runner=SimpleNamespace(_indicator_engine=SimpleNamespace(get_history=lambda s: [1] * 30)),
        order_manager=object(),
        active_trading_universe={},
        selected_ce=None,
        selected_pe=None,
    )
    out = await app._build_and_hydrate_live_basket_from_spot(ctx, spot_ltp=23701.0, configured_mode='LIVE')
    assert ctx.selected_ce and ctx.selected_pe
    assert ctx.active_trading_universe.get('selected_ce')
    assert out.get('selected_ce') == ctx.selected_ce
    await app._recompute_and_push_runtime_readiness(ctx, reason='test')
    assert ctx.selected_ce is not None and ctx.selected_pe is not None


@pytest.mark.asyncio
async def test_startup_hydration_preserves_requested_ssot_future(monkeypatch):
    basket = {
        'futures_symbol': 'NFO:NIFTY26MAYFUT',
        'selected_ce': 'NFO:NIFTY26JUN23700CE',
        'selected_pe': 'NFO:NIFTY26JUN23700PE',
        'atm_ce': 'NFO:NIFTY26JUN23700CE',
        'atm_pe': 'NFO:NIFTY26JUN23700PE',
        'atm_strike': 23700,
        'symbols': ['NSE:NIFTY', 'NFO:NIFTY26MAYFUT', 'NFO:NIFTY26JUN23700CE', 'NFO:NIFTY26JUN23700PE'],
        'option_symbols': ['NFO:NIFTY26JUN23700CE', 'NFO:NIFTY26JUN23700PE'],
    }
    monkeypatch.setattr(app, '_build_canonical_active_basket', lambda **kwargs: basket)
    monkeypatch.setattr(app, 'get_market_state', lambda: app.MarketState.CLOSED)
    ctx = SimpleNamespace(
        instrument_manager=_IM(),
        market_data_manager=_MDM(),
        broker_client=_Broker(),
        settings=SimpleNamespace(option_universe=SimpleNamespace(strike_step=50), execution_mode='LIVE'),
        strategy_runner=SimpleNamespace(_indicator_engine=SimpleNamespace(get_history=lambda s: [1] * 30)),
        order_manager=object(),
        active_trading_universe={},
        selected_ce=None,
        selected_pe=None,
    )
    out = await app._build_and_hydrate_live_basket_from_spot(ctx, spot_ltp=23701.0, configured_mode='LIVE')
    assert out.get("futures_symbol") == "NFO:NIFTY26MAYFUT"
    assert "NFO:NIFTY26MAYFUT" in (out.get("symbols") or [])


@pytest.mark.asyncio
async def test_startup_does_not_generate_calendar_future(monkeypatch):
    basket = {
        'futures_symbol': 'NFO:NIFTY26MAYFUT',
        'selected_ce': 'NFO:NIFTY26JUN23700CE',
        'selected_pe': 'NFO:NIFTY26JUN23700PE',
        'atm_ce': 'NFO:NIFTY26JUN23700CE',
        'atm_pe': 'NFO:NIFTY26JUN23700PE',
        'atm_strike': 23700,
        'symbols': ['NSE:NIFTY', 'NFO:NIFTY26MAYFUT', 'NFO:NIFTY26JUN23700CE', 'NFO:NIFTY26JUN23700PE'],
        'option_symbols': ['NFO:NIFTY26JUN23700CE', 'NFO:NIFTY26JUN23700PE'],
    }
    monkeypatch.setattr(app, '_get_current_nifty_futures_symbol', lambda: (_ for _ in ()).throw(AssertionError('calendar future generated')))
    monkeypatch.setattr(app, '_build_canonical_active_basket', lambda **kwargs: {**basket, 'futures_symbol': kwargs.get('futures_symbol') or ''})
    monkeypatch.setattr(app, 'get_market_state', lambda: app.MarketState.CLOSED)
    ctx = SimpleNamespace(
        instrument_manager=_IM(),
        market_data_manager=_MDM(),
        broker_client=_Broker(),
        settings=SimpleNamespace(option_universe=SimpleNamespace(strike_step=50), execution_mode='LIVE'),
        strategy_runner=SimpleNamespace(_indicator_engine=SimpleNamespace(get_history=lambda s: [1] * 30)),
        order_manager=object(),
        active_trading_universe={},
        selected_ce=None,
        selected_pe=None,
    )

    out = await app._build_and_hydrate_live_basket_from_spot(ctx, spot_ltp=23701.0, configured_mode='LIVE')

    assert out.get('futures_symbol') == 'NFO:NIFTY26JUNFUT'
    assert 'NFO:NIFTY26MAYFUT' not in (out.get('symbols') or [])

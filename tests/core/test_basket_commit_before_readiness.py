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


@pytest.mark.asyncio
async def test_hydration_not_ready_blocks_strategy_evaluation(monkeypatch):
    monkeypatch.setattr(app, 'get_market_state', lambda: app.MarketState.CLOSED)

    class MDM(_MDM):
        def hydrate_active_contract_basket(self, basket=None):
            return {
                "hard_ready": False,
                "missing": ["NFO:NIFTY26MAY23700CE:ohlc_insufficient"],
                "symbols": {},
            }

    calls = []
    ctx = SimpleNamespace(
        active_trading_universe={
            'selected_ce': 'NFO:NIFTY26MAY23700CE',
            'selected_pe': 'NFO:NIFTY26MAY23700PE',
            'atm_strike': 23700,
            'option_symbols': ['NFO:NIFTY26MAY23700CE', 'NFO:NIFTY26MAY23700PE'],
            'symbols': ['NSE:NIFTY', 'NFO:NIFTY26MAY23700CE', 'NFO:NIFTY26MAY23700PE'],
        },
        market_data_manager=MDM(),
        settings=SimpleNamespace(execution_mode='LIVE'),
        strategy_runner=SimpleNamespace(
            _indicator_engine=SimpleNamespace(get_history=lambda s: [1] * 30),
            set_runtime_readiness=lambda **kwargs: calls.append(kwargs),
            is_running=True,
        ),
        order_manager=object(),
        broker_client=object(),
        selected_ce=None,
        selected_pe=None,
    )
    await app._recompute_and_push_runtime_readiness(ctx, reason='test')
    assert ctx.evaluation_ready is False
    assert str(ctx.live_block_reason).startswith('ACTIVE_BASKET_HYDRATION_NOT_READY')
    assert calls[-1]['evaluation_ready'] is False


@pytest.mark.asyncio
async def test_hydration_ready_allows_strategy_evaluation(monkeypatch):
    monkeypatch.setattr(app, 'get_market_state', lambda: app.MarketState.CLOSED)

    class MDM(_MDM):
        def hydrate_active_contract_basket(self, basket=None):
            return {"hard_ready": True, "missing": [], "symbols": {}}

    calls = []
    ctx = SimpleNamespace(
        active_trading_universe={
            'selected_ce': 'NFO:NIFTY26MAY23700CE',
            'selected_pe': 'NFO:NIFTY26MAY23700PE',
            'atm_strike': 23700,
            'option_symbols': ['NFO:NIFTY26MAY23700CE', 'NFO:NIFTY26MAY23700PE'],
            'symbols': ['NSE:NIFTY', 'NFO:NIFTY26MAY23700CE', 'NFO:NIFTY26MAY23700PE'],
        },
        market_data_manager=MDM(),
        settings=SimpleNamespace(execution_mode='LIVE'),
        strategy_runner=SimpleNamespace(
            _indicator_engine=SimpleNamespace(get_history=lambda s: [1] * 30),
            set_runtime_readiness=lambda **kwargs: calls.append(kwargs),
            get_status=lambda: {"running": True},
        ),
        order_manager=object(),
        broker_client=object(),
        selected_ce=None,
        selected_pe=None,
    )
    await app._recompute_and_push_runtime_readiness(ctx, reason='test')
    assert ctx.evaluation_ready is True
    assert calls[-1]['evaluation_ready'] is True


@pytest.mark.asyncio
async def test_missing_hydration_method_blocks_live_readiness(monkeypatch, caplog):
    monkeypatch.delenv("ALLOW_MISSING_HYDRATION_GATE", raising=False)
    monkeypatch.setattr(app, 'get_market_state', lambda: app.MarketState.CLOSED)
    calls = []
    ctx = SimpleNamespace(
        active_trading_universe={
            'selected_ce': 'NFO:NIFTY26MAY23700CE',
            'selected_pe': 'NFO:NIFTY26MAY23700PE',
            'atm_strike': 23700,
            'option_symbols': ['NFO:NIFTY26MAY23700CE', 'NFO:NIFTY26MAY23700PE'],
            'symbols': ['NSE:NIFTY', 'NFO:NIFTY26MAY23700CE', 'NFO:NIFTY26MAY23700PE'],
        },
        market_data_manager=_MDM(),
        settings=SimpleNamespace(execution_mode='LIVE'),
        strategy_runner=SimpleNamespace(
            _indicator_engine=SimpleNamespace(get_history=lambda s: [1] * 30),
            set_runtime_readiness=lambda **kwargs: calls.append(kwargs),
            is_running=True,
        ),
        order_manager=object(),
        broker_client=object(),
        selected_ce=None,
        selected_pe=None,
    )
    await app._recompute_and_push_runtime_readiness(ctx, reason='test')
    assert ctx.evaluation_ready is False
    assert ctx.active_basket_hydration["hard_ready"] is False
    assert ctx.active_basket_hydration["missing"] == ["hydrate_active_contract_basket_missing"]
    assert "ACTIVE_BASKET_HYDRATION_METHOD_MISSING" in caplog.text
    assert calls[-1]['evaluation_ready'] is False


def test_botcontext_active_contract_basket_initialized():
    from nifty_scalper_bot.config.settings import Settings
    from nifty_scalper_bot.core.app import AppConfig, BotContext, RateLimiter

    ctx = BotContext(
        settings=Settings(),
        config=AppConfig(),
        rate_limiter=RateLimiter(),
        broker_client=None,
        websocket_client=None,
        websocket_manager=None,
        streamer=None,
        stream_supervisor=None,
        polling_fallback_streamer=None,
        message_bus=None,
    )

    assert ctx.active_contract_basket is None
    assert ctx.active_basket_hydration is None


@pytest.mark.asyncio
async def test_early_spot_basket_build_deferred_without_active_basket_attr_error(caplog):
    ctx = SimpleNamespace(
        instrument_manager=SimpleNamespace(is_loaded=lambda: False),
        market_data_manager=_MDM(),
        broker_client=_Broker(),
        settings=SimpleNamespace(option_universe=SimpleNamespace(strike_step=50), execution_mode='LIVE'),
        deferred_basket_retry_started=True,
        basket_build_lock=None,
        basket_build_in_progress=False,
        basket_build_last_started_mono=0.0,
        basket_build_last_completed_mono=0.0,
        basket_build_last_error=None,
    )

    result = await app._build_and_hydrate_live_basket_from_spot(ctx, spot_ltp=23701.0, configured_mode='LIVE')

    assert result["deferred"] is True
    assert result["reason"] == "instrument_manager_not_ready"
    assert "LIVE_BASKET_BUILD_DEFERRED" in caplog.text
    assert "LIVE_BASKET_BUILD_FAILED" not in caplog.text


@pytest.mark.asyncio
async def test_readiness_syncs_selected_option_runner_history_from_mdm_and_desired_subscription(caplog):
    ce = "NFO:NIFTY26JUN23900CE"
    pe = "NFO:NIFTY26JUN23900PE"
    base_bars = [
        {
            "timestamp": f"2026-05-01T09:{idx % 60:02d}:00+00:00",
            "open": 100.0 + idx,
            "high": 101.0 + idx,
            "low": 99.0 + idx,
            "close": 100.5 + idx,
            "volume": 100 + idx,
        }
        for idx in range(305)
    ]

    class _Runner:
        def __init__(self) -> None:
            self.counts = {ce: 18, pe: 18}
            self._running = True
            self.runtime_payloads = []
            self._indicator_engine = SimpleNamespace(get_history=lambda sym: [object()] * self.counts.get(sym, 0))

        def reseed_history_from_bars(self, sym, bars, source, min_bars):
            self.counts[sym] = max(len(bars), min_bars)
            return self.counts[sym]

        def set_runtime_readiness(self, **kwargs):
            self.runtime_payloads.append(kwargs)

    class _Mdm:
        _desired_tokens = {11, 12}
        _confirmed_subscriptions = set()
        _subscribed_tokens = set()
        _symbol_to_token = {ce: 11, pe: 12}

        def hydrate_active_contract_basket(self, basket):
            return {"hard_ready": True, "missing": [], "symbols": {}}

        def get_ohlc_bars(self, sym, limit=None):
            bars = list(base_bars)
            return bars[-limit:] if limit else bars

        async def hydrate_symbol_history(self, *args, **kwargs):
            raise AssertionError("MDM already has enough bars; broker hydration should not be called")

        def get_symbol_snapshot(self, sym):
            return SimpleNamespace(ltp=100.0, bid=99.0, ask=101.0, tradable_quote=True, depth_available=True, tick_age_s=0.1)

        def has_ws_tradable_quote(self, sym):
            return True

        def desired_tokens_snapshot(self):
            return [11, 12]

    runner = _Runner()
    ctx = SimpleNamespace(
        market_data_manager=_Mdm(),
        strategy_runner=runner,
        settings=SimpleNamespace(execution_mode="SHADOW"),
        broker_client=object(),
        order_manager=object(),
        selected_ce=ce,
        selected_pe=pe,
        active_symbol_tokens={ce: 11, "NIFTY26JUN23900CE": 11, pe: 12, "NIFTY26JUN23900PE": 12},
        active_trading_universe={
            "spot_symbol": "NSE:NIFTY",
            "selected_ce": ce,
            "selected_pe": pe,
            "option_symbols": [ce, pe],
            "all_symbols": ["NSE:NIFTY", ce, pe],
            "token_by_symbol": {ce: 11, pe: 12},
            "all_tokens": [256265, 11, 12],
        },
        active_contract_basket=None,
        data_hard_ready=False,
        evaluation_ready=False,
        live_orders_armed=False,
        trading_ready=False,
    )

    with caplog.at_level("INFO"):
        await app._recompute_and_push_runtime_readiness(ctx, reason="test")

    assert runner.counts[ce] >= 30
    assert runner.counts[pe] >= 30
    assert runner.runtime_payloads[-1]["evaluation_ready"] is True
    assert "ce_eval_bars_missing" not in str(ctx.live_block_reason)
    assert "pe_eval_bars_missing" not in str(ctx.live_block_reason)
    assert "SELECTED_OPTION_RUNNER_SYNC_FROM_MDM" in caplog.text
    assert "SELECTED_OPTION_RESEED_FAILED" not in caplog.text

from types import SimpleNamespace
import pytest
from nifty_scalper_bot.core import app

class Runner:
    def __init__(self):
        self._indicator_engine = SimpleNamespace(get_history=lambda s: self.hist.get(s, []))
        self.hist = {}
    def sync_history_from_mdm(self, sym, *, required_bars, reason, role=None, request_if_short=False):
        self.hist[sym] = [object()] * required_bars
        return SimpleNamespace(runner_bars=required_bars, indicator_bars=required_bars, success=True, failure_reason=None)
    def set_runtime_readiness(self, **kwargs):
        pass

class MDM:
    def __init__(self):
        self.store = {'NSE:NIFTY':[1]}
    def get_ohlc_bars(self, s, limit=None):
        v=self.store.get(s,[])
        return v[-limit:] if limit else v
    def get_symbol_snapshot(self,s):
        return {'ltp':1}
    async def ensure_history(self, sym, **kwargs):
        bars = kwargs.get('target_bars') or kwargs.get('required_bars') or 30
        self.store[sym]=[{'open':1,'high':1,'low':1,'close':1,'volume':1} for _ in range(bars)]
        return SimpleNamespace(minimum_ready=True, target_ready=True, fetched_rows=bars, accepted_rows=bars, failure_reason=None)
    def update_hydration_status(self,s,b):
        pass

@pytest.mark.asyncio
async def test_selected_options_force_hydration_reaches_exec_ready():
    mdm=MDM(); r=Runner()
    ctx=SimpleNamespace(settings=SimpleNamespace(execution_mode='LIVE'),market_data_manager=mdm,strategy_runner=r,broker_client=object(),order_manager=object(),active_trading_universe={'spot_symbol':'NSE:NIFTY','selected_ce':'NFO:CE','selected_pe':'NFO:PE','option_symbols':['NFO:CE','NFO:PE']})
    await app._recompute_and_push_runtime_readiness(ctx, reason='t')
    assert len(mdm.get_ohlc_bars('NFO:CE')) >= 5
    assert len(r.hist.get('NFO:CE', [])) >= 5

@pytest.mark.asyncio
async def test_fresh_basket_selection_overrides_stale_ctx_selected_symbols():
    mdm=MDM(); r=Runner()
    ctx=SimpleNamespace(settings=SimpleNamespace(execution_mode='LIVE'),market_data_manager=mdm,strategy_runner=r,broker_client=object(),order_manager=object(),selected_ce='OLDCE',selected_pe='OLDPE',active_trading_universe={'spot_symbol':'NSE:NIFTY','selected_ce':'NEWCE','selected_pe':'NEWPE','option_symbols':['NEWCE','NEWPE']})
    await app._recompute_and_push_runtime_readiness(ctx, reason='t')
    assert ctx.selected_ce=='NEWCE' and ctx.selected_pe=='NEWPE'


def _indicator_sync_ctx(bars: list[int]) -> SimpleNamespace:
    symbol = "NFO:NIFTY26NOV24900CE"
    return SimpleNamespace(
        strategy_runner=SimpleNamespace(
            _required_candles=30,
            _active_symbols={symbol},
        ),
        indicator_engine=SimpleNamespace(
            has_min_bars=lambda sym, required: sym == symbol and bars[0] >= required,
            get_history=lambda sym: [object()] * bars[0] if sym == symbol else [],
        ),
        data_hub=SimpleNamespace(indicators_ready=False),
        market_regime_manager=SimpleNamespace(indicators_ready=False),
        live_orders_armed=False,
    )


def test_live_indicator_sync_rearms_only_after_real_history_and_hard_readiness():
    bars = [25]
    ctx = _indicator_sync_ctx(bars)
    assert app._sync_runtime_indicator_readiness(ctx, data_hard_ready=True) is False

    bars[0] = 30
    assert app._sync_runtime_indicator_readiness(ctx, data_hard_ready=True) is True
    assert ctx.data_hub.indicators_ready is True
    assert ctx.market_regime_manager.indicators_ready is True
    assert ctx.live_orders_armed is False

    # A failed canonical readiness gate must revoke the global indicator flag.
    assert app._sync_runtime_indicator_readiness(ctx, data_hard_ready=False) is False
    assert ctx.data_hub.indicators_ready is False
    assert ctx.market_regime_manager.indicators_ready is False

    # Regressing indicator history also revokes readiness.
    bars[0] = 29
    assert app._sync_runtime_indicator_readiness(ctx, data_hard_ready=True) is False


def test_live_indicator_sync_fails_closed_when_runner_missing():
    ctx = SimpleNamespace(
        strategy_runner=None,
        indicator_engine=None,
        data_hub=SimpleNamespace(indicators_ready=True),
    )
    assert app._sync_runtime_indicator_readiness(ctx, data_hard_ready=True) is False
    assert ctx.data_hub.indicators_ready is False

from types import SimpleNamespace
import pytest
from nifty_scalper_bot.core import app

class Runner:
    def __init__(self):
        self._indicator_engine = SimpleNamespace(get_history=lambda s: self.hist.get(s, []))
        self.hist = {}
    def ingest_historical_bar(self, bar):
        self.hist.setdefault(bar['symbol'], []).append(bar)
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
    async def hydrate_symbol_history(self, sym, **kwargs):
        self.store[sym]=[{'open':1,'high':1,'low':1,'close':1,'volume':1} for _ in range(10)]
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

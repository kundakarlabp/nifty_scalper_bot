from types import SimpleNamespace
import pytest
from nifty_scalper_bot.core import app


class MDM:
    def get_ohlc_bars(self, s, limit=None): return [1] * 30
    def get_symbol_snapshot(self, s): return SimpleNamespace(ltp=100.0, bid=99.0, ask=101.0, tradable_quote=True, tick_age_s=0.1)
    async def hydrate_symbol_history(self, *args, **kwargs): return []


@pytest.mark.asyncio
async def test_readiness_backfills_ctx_selected_from_basket(monkeypatch):
    monkeypatch.setattr(app, 'get_market_state', lambda: app.MarketState.CLOSED)
    ctx = SimpleNamespace(
        settings=SimpleNamespace(execution_mode='LIVE'), market_data_manager=MDM(),
        strategy_runner=SimpleNamespace(_indicator_engine=SimpleNamespace(get_history=lambda s: [1] * 30)),
        broker_client=object(), order_manager=object(), selected_ce=None, selected_pe=None,
        active_trading_universe={'selected_ce': 'CE', 'selected_pe': 'PE', 'option_symbols': ['CE', 'PE'], 'spot_symbol': 'NSE:NIFTY'},
    )
    await app._recompute_and_push_runtime_readiness(ctx, reason='t')
    assert ctx.selected_ce == 'CE'
    assert ctx.selected_pe == 'PE'

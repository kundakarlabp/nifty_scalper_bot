from types import SimpleNamespace

import pytest

from nifty_scalper_bot.core import app
from nifty_scalper_bot.utils.market_hours import MarketState


class Runner:
    def __init__(self) -> None:
        self._running = True
        self._indicator_engine = SimpleNamespace(
            get_history=lambda _s: [1] * 50,
        )

    def set_runtime_readiness(self, **kwargs) -> None:
        self.last_runtime = kwargs


class MDM:
    def get_symbol_snapshot(self, _s):
        return SimpleNamespace(ltp=100.0, bid=99.0, ask=101.0, tradable_quote=True, depth_available=True, tick_age_s=0.2)

    def get_ohlc_bars(self, _s):
        return [1] * 50


@pytest.mark.asyncio
async def test_off_market_live_orders_are_not_armed(monkeypatch) -> None:
    async def _noop_hydrate(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(app, 'get_market_state', lambda: MarketState.CLOSED)
    monkeypatch.setattr(app, '_ensure_selected_options_hydrated', _noop_hydrate)
    ctx = SimpleNamespace(
        settings=SimpleNamespace(execution_mode='LIVE'),
        market_data_manager=MDM(),
        strategy_runner=Runner(),
        broker_client=object(),
        order_manager=object(),
        selected_ce='NFO:NIFTY2460020000CE',
        selected_pe='NFO:NIFTY2460020000PE',
        active_trading_universe={
            'spot_symbol': 'NSE:NIFTY',
            'selected_ce': 'NFO:NIFTY2460020000CE',
            'selected_pe': 'NFO:NIFTY2460020000PE',
            'option_symbols': ['NFO:NIFTY2460020000CE', 'NFO:NIFTY2460020000PE'],
        },
    )
    await app._recompute_and_push_runtime_readiness(ctx, reason='test')
    assert ctx.live_orders_armed is False
    assert 'market_closed' in str(ctx.live_block_reason)

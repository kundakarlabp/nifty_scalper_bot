from __future__ import annotations

from types import SimpleNamespace
from datetime import UTC, datetime

import pytest

from nifty_scalper_bot.core import app


class Snap:
    def __init__(self, ltp: float, age: float, tradable: bool = False, bid: float | None = None, ask: float | None = None, depth: bool = False):
        self.ltp = ltp
        self.tick_age_s = age
        self.tradable_quote = tradable
        self.bid = bid
        self.ask = ask
        self.depth_available = depth


class Runner:
    def __init__(self) -> None:
        self.calls = []
        self._running = True
        self._indicator_engine = SimpleNamespace(get_history=lambda s: list(range(60)))

    def set_runtime_readiness(self, **kwargs):
        self.calls.append(kwargs)


def make_ctx(mdm, live: bool = True):
    return SimpleNamespace(
        settings=SimpleNamespace(execution_mode='LIVE' if live else 'PAPER'),
        market_data_manager=mdm,
        strategy_runner=Runner(),
        broker_client=object(),
        order_manager=object(),
        active_trading_universe={
            'spot_symbol': 'NSE:NIFTY',
            'selected_ce': 'NFO:CE',
            'selected_pe': 'NFO:PE',
            'option_symbols': ['NFO:CE', 'NFO:PE'],
            'atm_strike': 23400,
        },
        execution_locked_symbols=set(),
        execution_lock_timestamps={},
    )


def _fresh_tick_ts() -> dict[str, str]:
    now_iso = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    return {'NFO:CE': now_iso, 'NFO:PE': now_iso}


@pytest.mark.asyncio
async def test_runtime_readiness_uses_live_tick_proof_when_confirmed_subscription_missing() -> None:
    snaps = {'NSE:NIFTY': Snap(25000, 2), 'NFO:CE': Snap(100, 2), 'NFO:PE': Snap(110, 3)}
    mdm = SimpleNamespace(get_symbol_snapshot=lambda s: snaps.get(s), get_ohlc_bars=lambda s, **k: list(range(60)), _confirmed_subscriptions=set(), _symbol_to_token={'NFO:CE': 1, 'NFO:PE': 2}, _last_tick_ts=_fresh_tick_ts())
    ctx = make_ctx(mdm)
    await app._recompute_and_push_runtime_readiness(ctx, reason='test')
    assert ctx.data_hard_ready is True
    assert ctx.evaluation_ready is True
    assert ctx.strategy_runner.calls[-1]['data_hard_ready'] is True
    assert ctx.strategy_runner.calls[-1]['evaluation_ready'] is True
    assert str(ctx.live_block_reason).startswith('execution_not_armed:') or ctx.live_orders_armed


@pytest.mark.asyncio
async def test_live_orders_not_armed_without_tradable_quote() -> None:
    snaps = {'NSE:NIFTY': Snap(25000, 2), 'NFO:CE': Snap(100, 2), 'NFO:PE': Snap(110, 2)}
    mdm = SimpleNamespace(get_symbol_snapshot=lambda s: snaps.get(s), get_ohlc_bars=lambda s, **k: list(range(60)), _confirmed_subscriptions=set(), _symbol_to_token={'NFO:CE': 1, 'NFO:PE': 2}, _last_tick_ts=_fresh_tick_ts())
    ctx = make_ctx(mdm)
    await app._recompute_and_push_runtime_readiness(ctx, reason='test')
    assert ctx.data_hard_ready is True
    assert ctx.evaluation_ready is True
    assert ctx.live_orders_armed is False
    assert str(ctx.live_block_reason).startswith('execution_not_armed:')


@pytest.mark.asyncio
async def test_live_orders_armed_with_fresh_ltp_bars_and_tradable_depth(monkeypatch) -> None:
    monkeypatch.setattr(app, 'get_market_state', lambda: app.MarketState.OPEN)
    snaps = {'NSE:NIFTY': Snap(25000, 2, True, 24999, 25001, True), 'NFO:CE': Snap(100, 2, True, 99.95, 100.05, True), 'NFO:PE': Snap(110, 2, True, 109.95, 110.05, True)}
    mdm = SimpleNamespace(get_symbol_snapshot=lambda s: snaps.get(s), get_ohlc_bars=lambda s, **k: list(range(60)), has_ws_tradable_quote=lambda s: True, _confirmed_subscriptions=set(), _symbol_to_token={'NFO:CE': 1, 'NFO:PE': 2}, _last_tick_ts=_fresh_tick_ts())
    ctx = make_ctx(mdm)
    await app._recompute_and_push_runtime_readiness(ctx, reason='test')
    assert ctx.data_hard_ready is True
    assert ctx.evaluation_ready is True
    assert ctx.live_orders_armed is True
    assert ctx.trading_ready is True


@pytest.mark.asyncio
async def test_ensure_selected_options_hydrated_handles_runner_reseed_failure() -> None:
    symbol = 'NFO:CE'

    class RunnerFailing:
        def __init__(self) -> None:
            self._indicator_engine = SimpleNamespace(get_history=lambda _s: list(range(5)))

        def reseed_history_from_bars(self, *args, **kwargs):
            raise RuntimeError('boom')

    class MdmStub:
        async def hydrate_symbol_history(self, *args, **kwargs) -> None:
            return None

        def get_ohlc_bars(self, _sym: str, limit: int | None = None):
            bars = [
                {'start': '2026-05-01T03:45:00+00:00', 'open': 100, 'high': 101, 'low': 99, 'close': 100.5},
            ]
            return bars[: limit or len(bars)]

    ctx = SimpleNamespace(market_data_manager=MdmStub(), strategy_runner=RunnerFailing())

    result = await app._ensure_selected_options_hydrated(ctx, symbol, None, 2, 'test')
    assert result[symbol]['ready'] is False

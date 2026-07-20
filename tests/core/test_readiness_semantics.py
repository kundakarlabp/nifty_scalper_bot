from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.core import app


class _MDM:
    def hydrate_active_contract_basket(self, basket=None):
        return {"hard_ready": True, "missing": [], "symbols": {}}

    def get_symbol_snapshot(self, symbol):
        return SimpleNamespace(
            ltp=100.0,
            tick_age_s=1.0,
            bid=99.0,
            ask=101.0,
            tradable_quote=True,
            depth_available=True,
        )

    def has_ws_tradable_quote(self, symbols):
        return True

    def desired_tokens_snapshot(self):
        return [11, 12]

    def get_ohlc_bars(self, symbol):
        end = datetime.now(UTC).replace(second=0, microsecond=0)
        return [
            {"timestamp": end - timedelta(minutes=30 - idx), "close": 100.0}
            for idx in range(30)
        ]

    def hydrate_symbol_history(self, symbol, **kwargs):
        return self.get_ohlc_bars(symbol)

    _subscribed_tokens = {11, 12}
    _confirmed_subscriptions = {11, 12}
    _symbol_to_token = {"NFO:NIFTY26JUN25000CE": 11, "NFO:NIFTY26JUN25000PE": 12}


@pytest.mark.asyncio
async def test_off_market_data_ready_splits_execution_arming(monkeypatch) -> None:
    monkeypatch.setattr(app, "get_market_state", lambda: app.MarketState.CLOSED)
    ce = "NFO:NIFTY26JUN25000CE"
    pe = "NFO:NIFTY26JUN25000PE"
    calls = []
    ctx = SimpleNamespace(
        active_trading_universe={
            "selected_ce": ce,
            "selected_pe": pe,
            "atm_strike": 25000,
            "option_symbols": [ce, pe],
            "token_by_symbol": {ce: 11, pe: 12},
        },
        active_symbol_tokens={ce: 11, pe: 12},
        selected_ce=ce,
        selected_pe=pe,
        market_data_manager=_MDM(),
        settings=SimpleNamespace(execution_mode="LIVE"),
        strategy_runner=SimpleNamespace(
            get_status=lambda: {"running": True},
            _indicator_engine=SimpleNamespace(
                get_history=lambda s: ctx.market_data_manager.get_ohlc_bars(s)
            ),
            set_runtime_readiness=lambda **kw: calls.append(kw),
        ),
        order_manager=object(),
        broker_client=object(),
    )

    await app._recompute_and_push_runtime_readiness(ctx, reason="unit_test")

    assert ctx.data_ready is True
    assert ctx.strategy_evaluation_ready is True
    assert ctx.trading_ready is True
    assert ctx.execution_armed is False
    assert ctx.live_orders_armed is False
    assert ctx.execution_block_reason == "market_closed"
    assert calls[-1]["evaluation_ready"] is True

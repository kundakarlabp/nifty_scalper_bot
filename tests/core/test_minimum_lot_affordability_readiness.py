from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.core import app

CE = "NFO:NIFTY26AUG25000CE"
PE = "NFO:NIFTY26AUG25000PE"


class _MDM:
    _subscribed_tokens = {11, 12}
    _confirmed_subscriptions = {11, 12}
    _symbol_to_token = {CE: 11, PE: 12}

    def __init__(self, asks: dict[str, float]):
        self.asks = asks
        self.pipeline_overloaded = False

    def hydrate_active_contract_basket(self, basket=None):
        return {"hard_ready": True, "missing": [], "symbols": {}}

    def get_symbol_snapshot(self, symbol):
        ask = self.asks.get(symbol, 100.0)
        return SimpleNamespace(
            ltp=ask - 0.5,
            tick_age_s=1.0,
            bid=ask - 1.0,
            ask=ask,
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
            {
                "timestamp": end - timedelta(minutes=30 - idx),
                "close": 100.0,
            }
            for idx in range(30)
        ]

    def hydrate_symbol_history(self, symbol, **kwargs):
        return self.get_ohlc_bars(symbol)


class _OrderManager:
    _margin_factor = 1.0
    _margin_buffer = 0.8

    def resolve_lot_size(self, symbol: str) -> int:
        return 65


class _Hub:
    def __init__(self, balance: float):
        self.balance = balance

    def get_available_balance(self, force=False):
        return self.balance


def _context(asks: dict[str, float], balance: float):
    calls = []
    ctx = SimpleNamespace(
        active_trading_universe={
            "selected_ce": CE,
            "selected_pe": PE,
            "atm_strike": 25000,
            "option_symbols": [CE, PE],
            "token_by_symbol": {CE: 11, PE: 12},
        },
        active_symbol_tokens={CE: 11, PE: 12},
        selected_ce=CE,
        selected_pe=PE,
        market_data_manager=_MDM(asks),
        data_hub=_Hub(balance),
        settings=SimpleNamespace(execution_mode="LIVE"),
        strategy_runner=SimpleNamespace(
            get_status=lambda: {"running": True},
            _indicator_engine=SimpleNamespace(
                get_history=lambda s: ctx.market_data_manager.get_ohlc_bars(s)
            ),
            set_runtime_readiness=lambda **kw: calls.append(kw),
            has_datahub_subscription=lambda *args, **kwargs: True,
        ),
        order_manager=_OrderManager(),
        broker_client=object(),
        broker_balance_valid=True,
        last_valid_broker_balance=balance,
        position_reconciliation_completed=True,
        position_reconciliation_completed_at=datetime.now(UTC),
        position_reconciliation_failed=False,
    )
    return ctx, calls


@pytest.mark.asyncio
async def test_both_selected_contracts_unaffordable_disarm_live_orders(
    monkeypatch,
) -> None:
    monkeypatch.setattr(app, "get_market_state", lambda: app.MarketState.OPEN)
    ctx, calls = _context({CE: 172.0, PE: 180.0}, 10_000.0)

    await app._recompute_and_push_runtime_readiness(ctx, reason="minimum_lot_test")

    assert ctx.evaluation_ready is True
    assert ctx.execution_capacity_ready is False
    assert ctx.live_orders_armed is False
    assert ctx.live_block_reason == "execution_not_armed:minimum_lot_unaffordable"
    assert ctx.execution_ready_by_symbol == {CE: False, PE: False}
    assert calls[-1]["live_orders_armed"] is False


@pytest.mark.asyncio
async def test_one_affordable_side_keeps_global_arming_and_side_gate(
    monkeypatch,
) -> None:
    monkeypatch.setattr(app, "get_market_state", lambda: app.MarketState.OPEN)
    ctx, calls = _context({CE: 100.0, PE: 180.0}, 10_000.0)

    await app._recompute_and_push_runtime_readiness(ctx, reason="minimum_lot_test")

    assert ctx.execution_capacity_ready is True
    assert ctx.live_orders_armed is True
    assert ctx.execution_ready_by_symbol == {CE: True, PE: False}
    assert ctx.live_block_reason is None
    assert calls[-1]["execution_ready_by_symbol"] == {CE: True, PE: False}
    assert calls[-1]["live_orders_armed"] is True


@pytest.mark.asyncio
async def test_pipeline_overload_disarms_canonical_runtime_readiness(
    monkeypatch,
) -> None:
    monkeypatch.setattr(app, "get_market_state", lambda: app.MarketState.OPEN)
    ctx, calls = _context({CE: 100.0, PE: 110.0}, 10_000.0)
    ctx.market_data_manager.pipeline_overloaded = True

    await app._recompute_and_push_runtime_readiness(ctx, reason="overload_test")

    assert ctx.evaluation_ready is True
    assert ctx.live_orders_armed is False
    assert ctx.live_block_reason == "execution_not_armed:data_pipeline_overloaded"
    assert calls[-1]["live_orders_armed"] is False


@pytest.mark.asyncio
async def test_authoritative_risk_breaker_disarms_runtime_readiness(
    monkeypatch,
) -> None:
    monkeypatch.setattr(app, "get_market_state", lambda: app.MarketState.OPEN)
    ctx, calls = _context({CE: 100.0, PE: 110.0}, 10_000.0)
    ctx.risk_manager = SimpleNamespace(
        is_circuit_breaker_tripped=lambda: (True, "Consecutive loss limit reached")
    )

    await app._recompute_and_push_runtime_readiness(ctx, reason="risk_breaker_test")

    assert ctx.live_orders_armed is False
    assert ctx.live_block_reason == "execution_not_armed:risk_halt"
    assert calls[-1]["live_orders_armed"] is False

    ctx.risk_manager.is_circuit_breaker_tripped = lambda: (False, "")
    await app._recompute_and_push_runtime_readiness(ctx, reason="risk_reset_test")

    assert ctx.risk_halt is False
    assert ctx.live_orders_armed is True


@pytest.mark.asyncio
async def test_risk_breaker_read_failure_keeps_runtime_disarmed(monkeypatch) -> None:
    monkeypatch.setattr(app, "get_market_state", lambda: app.MarketState.OPEN)
    ctx, _calls = _context({CE: 100.0, PE: 110.0}, 10_000.0)

    def _failed_breaker_read():
        raise RuntimeError("risk state unavailable")

    ctx.risk_manager = SimpleNamespace(is_circuit_breaker_tripped=_failed_breaker_read)

    await app._recompute_and_push_runtime_readiness(ctx, reason="risk_error_test")

    assert ctx.risk_halt is True
    assert ctx.live_orders_armed is False
    assert ctx.live_block_reason == "execution_not_armed:risk_halt"

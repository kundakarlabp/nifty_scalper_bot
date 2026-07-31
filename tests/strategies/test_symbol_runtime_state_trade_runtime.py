from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from nifty_scalper_bot.core.message_bus import MessageBus
from nifty_scalper_bot.strategies.runner import (
    StrategyRunner,
    StrategyRunnerConfig,
    SymbolRuntimeState,
)


class _StubOrderManager:
    def place_order(self, **_kwargs):
        raise AssertionError("order placement is not part of this test")

    def place_reduce_only_exit(self, *_args, **_kwargs):
        raise AssertionError("exit placement is not part of this test")

    def consume_skip_reason(self):
        return None

    def get_order(self, _order_id):
        return None


class _StubPositionManager:
    def get_active_contract(self, _underlying):
        return None

    def is_flat(self, _symbol):
        return True

    def clear_active_contract(self, _underlying):
        return None

    def get_position(self, _symbol):
        return None

    def set_active_contract(self, _underlying, _contract):
        return None


def _build_runner(
    strategy_profile: dict[str, object] | None = None,
) -> StrategyRunner:
    return StrategyRunner(
        market_data_manager=SimpleNamespace(),
        indicator_engine=SimpleNamespace(),
        strategy_manager=SimpleNamespace(),
        risk_manager=SimpleNamespace(),
        order_manager=_StubOrderManager(),
        position_manager=_StubPositionManager(),
        message_bus=MessageBus(),
        config=StrategyRunnerConfig(max_trade_history=5),
        strategy_profile=strategy_profile,
    )


def test_runner_uses_derived_strategy_profile_version() -> None:
    runner = _build_runner(
        {"schema_version": 1, "version": "production-v1-abc123def456"}
    )

    assert (
        runner._build_info["strategy_profile_version"]
        == "production-v1-abc123def456"
    )
    assert runner._strategy_profile["schema_version"] == 1


def test_symbol_runtime_state_defaults_without_last_trade_at() -> None:
    state = SymbolRuntimeState(symbol="NFO:NIFTY26JUN24000CE", history_limit=5)

    assert state.last_trade_at is None
    assert state.last_order_id is None
    assert state.last_trade_symbol is None
    assert state.snapshot()["last_trade_at"] is None


def test_symbol_runtime_state_cooldown_handles_missing_and_none_last_trade_at() -> None:
    state = SymbolRuntimeState(symbol="NFO:NIFTY26JUN24000CE", history_limit=5)

    assert state.trade_cooldown_remaining(now=100.0, cooldown_seconds=10.0) == 0.0

    # Simulate an old/unpickled object whose slots predate last_trade_at.
    old_state = object.__new__(SymbolRuntimeState)
    assert old_state.trade_cooldown_remaining(now=100.0, cooldown_seconds=10.0) == 0.0


def test_symbol_runtime_state_setting_last_trade_at_controls_cooldown() -> None:
    state = SymbolRuntimeState(symbol="NFO:NIFTY26JUN24000CE", history_limit=5)
    state.last_trade_at = 95.0

    assert state.trade_cooldown_remaining(now=100.0, cooldown_seconds=10.0) == 5.0
    assert state.trade_cooldown_remaining(now=120.0, cooldown_seconds=10.0) == 0.0


def test_symbol_runtime_state_invalid_or_future_last_trade_at_is_safe() -> None:
    state = SymbolRuntimeState(symbol="NFO:NIFTY26JUN24000CE", history_limit=5)
    state.last_trade_at = -1.0
    assert state.trade_cooldown_remaining(now=100.0, cooldown_seconds=10.0) == 0.0

    state.last_trade_at = 105.0
    assert state.trade_cooldown_remaining(now=100.0, cooldown_seconds=10.0) == 10.0


def test_runner_restore_old_trade_payload_sets_epoch_last_trade_at() -> None:
    runner = _build_runner()
    timestamp = datetime(2026, 6, 10, 4, 0, tzinfo=timezone.utc)

    runner.restore_trades(
        [
            {
                "symbol": "NFO:NIFTY26JUN24000CE",
                "timestamp": timestamp.isoformat(),
                "action": "BUY",
                "quantity": 75,
                "price": 100.5,
                "status": "FILLED",
                "order_id": "OID-1",
            }
        ]
    )

    state = runner._symbol_state["NFO:NIFTY26JUN24000CE"]
    assert state.last_trade_at == timestamp.timestamp()
    assert state.last_order_id == "OID-1"
    assert state.last_trade_symbol == "NFO:NIFTY26JUN24000CE"
    assert state.snapshot()["last_trade_at"] == timestamp.timestamp()


def test_runner_initialization_path_has_last_trade_at_field() -> None:
    runner = _build_runner()

    runner.add_symbol("NFO:NIFTY26JUN24000CE")

    assert runner._symbol_state["NFO:NIFTY26JUN24000CE"].last_trade_at is None

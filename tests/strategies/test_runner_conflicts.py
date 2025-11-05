from __future__ import annotations

from datetime import datetime, timezone
import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from nifty_scalper_bot.options.strike_selector import SelectedContract
from nifty_scalper_bot.strategies.runner import (
    StrategyRunner,
    StrategyRunnerConfig,
    SymbolState,
)
from nifty_scalper_bot.strategies.signal_generator import Signal


class _DummyPositionManager:
    def __init__(self, active_contract: Any) -> None:
        self._active = active_contract
        self.set_calls: list[tuple[str, SelectedContract]] = []
        self.cleared: list[str] = []

    def get_active_contract(self, underlying: str) -> Any:
        return self._active

    def is_flat(self, _symbol: str) -> bool:
        return False

    def clear_active_contract(self, underlying: str) -> None:
        self.cleared.append(underlying)

    def get_position(self, _symbol: str) -> Any:
        return None

    def set_active_contract(self, underlying: str, contract: SelectedContract) -> None:
        self.set_calls.append((underlying, contract))
        self._active = contract


class _DummySelector:
    def __init__(self, selection: SelectedContract | None) -> None:
        self.selection = selection
        self.calls: list[tuple[str, str, str, float]] = []
        self.registered: list[tuple[str, SelectedContract]] = []
        self.settings = SimpleNamespace(long_only=True, legacy_side_to_type=False)

    def select_contract(
        self,
        *,
        underlying: str,
        side: str,
        option_type: str,
        underlying_price: float,
        **_: object,
    ) -> SelectedContract | None:
        self.calls.append((underlying, side, option_type, underlying_price))
        return self.selection

    def register_open(self, underlying: str, contract: SelectedContract) -> None:
        self.registered.append((underlying, contract))


def _make_runner(
    *,
    position_manager: _DummyPositionManager,
    selector: _DummySelector,
) -> tuple[StrategyRunner, MagicMock, MagicMock]:
    market_data = MagicMock()
    indicator = MagicMock()
    strategy_manager = MagicMock()
    risk_manager = MagicMock()
    order_manager = MagicMock()
    data_hub = None

    cfg = StrategyRunnerConfig(max_trade_history=5)
    runner = StrategyRunner(
        market_data_manager=market_data,
        indicator_engine=indicator,
        strategy_manager=strategy_manager,
        risk_manager=risk_manager,
        order_manager=order_manager,
        position_manager=position_manager,  # type: ignore[arg-type]
        config=cfg,
        data_hub=data_hub,
        strike_selector=selector,
    )
    runner._symbol_state["NIFTY"] = SymbolState(symbol="NIFTY", history_limit=5)
    runner._symbol_state["NIFTY"].active = True
    runner._symbol_state["NIFTY"].strategy_data["last_signal"] = {}
    return runner, risk_manager, order_manager


def _active_contract(option_type: str) -> Any:
    return SimpleNamespace(
        symbol=f"NIFTY25O2025450{option_type}",
        option_type=option_type,
        strike=25450.0,
        expiry=datetime(2025, 10, 30, tzinfo=timezone.utc),
    )


def _make_signal(
    action: str,
    *,
    option_type: str = "CE",
    metadata: dict[str, Any] | None = None,
) -> Signal:
    payload: dict[str, Any] = {"option_type": option_type}
    if metadata:
        payload.update(metadata)
    return Signal(
        action=action,
        symbol="NIFTY",
        quantity=75,
        confidence=0.8,
        reason="test",
        stop_loss=100.0,
        take_profit=150.0,
        metadata=payload,
    )


def test_runner_blocks_conflicting_contract_without_hedge(monkeypatch, caplog) -> None:
    monkeypatch.delenv("NSB__ALLOW_HEDGE_ENTRIES", raising=False)
    position_manager = _DummyPositionManager(_active_contract("PE"))
    selector = _DummySelector(None)
    runner, risk_manager, order_manager = _make_runner(
        position_manager=position_manager,
        selector=selector,
    )
    risk_manager.suggest_position_size.return_value = 75
    risk_manager.validate_new_position.return_value = (True, "")
    caplog.set_level(logging.INFO)

    runner._handle_signal(
        _make_signal("BUY"),
        price=120.0,
        timestamp=datetime.now(timezone.utc),
    )

    order_manager.place_order.assert_not_called()
    assert any(
        "conflicting_active_contract_blocked" in record.message
        for record in caplog.records
    )
    history = list(runner._symbol_state["NIFTY"].trade_history)
    assert history
    assert history[-1].status == "blocked"
    assert history[-1].reason == "conflicting_active_contract"


def test_runner_allows_conflicting_contract_with_hedge(monkeypatch, caplog) -> None:
    monkeypatch.setenv("NSB__ALLOW_HEDGE_ENTRIES", "true")
    position_manager = _DummyPositionManager(_active_contract("PE"))
    selection = SelectedContract(
        symbol="NIFTY25O2025450CE",
        option_type="CE",
        strike=25450.0,
        expiry=datetime(2025, 10, 30, tzinfo=timezone.utc),
        ltp=112.0,
        delta=None,
        metadata={},
    )
    selector = _DummySelector(selection)
    runner, risk_manager, order_manager = _make_runner(
        position_manager=position_manager,
        selector=selector,
    )
    risk_manager.suggest_position_size.return_value = 75
    risk_manager.validate_new_position.return_value = (True, "")
    order_manager.place_order.return_value = "OID-1"
    caplog.set_level(logging.INFO)

    runner._handle_signal(
        _make_signal("BUY"),
        price=118.0,
        timestamp=datetime.now(timezone.utc),
    )

    order_manager.place_order.assert_called_once()
    assert any(
        "conflicting_active_contract_allowed" in record.message
        for record in caplog.records
    )
    assert position_manager.set_calls == []
    history = list(runner._symbol_state["NIFTY"].trade_history)
    assert history
    assert history[-1].status == "submitted"

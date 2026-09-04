from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.config.settings import RiskSettings, _build_risk_settings
from nifty_scalper_bot.execution import BracketManager
from nifty_scalper_bot.execution.order_manager import OrderManager, OrderType
from nifty_scalper_bot.risk.entry_guard_patch import _daily_limit_block_reason


CE = "NFO:NIFTY2690823950CE"


def test_entry_at_single_position_capacity_is_rejected() -> None:
    manager = SimpleNamespace(
        settings=SimpleNamespace(max_trades_per_day=0, max_open_positions=1),
        position_manager=SimpleNamespace(
            get_open_positions=lambda: [SimpleNamespace(symbol=CE, quantity=65)]
        ),
    )

    blocker = _daily_limit_block_reason(manager)

    assert blocker == (
        "max_open_positions breached: 1/1",
        "MAX_OPEN:1/1",
    )


def test_entry_below_single_position_capacity_remains_allowed() -> None:
    manager = SimpleNamespace(
        settings=SimpleNamespace(max_trades_per_day=0, max_open_positions=1),
        position_manager=SimpleNamespace(get_open_positions=lambda: []),
    )

    assert _daily_limit_block_reason(manager) is None


def test_risk_settings_default_matches_safe_live_policy() -> None:
    assert RiskSettings().per_trade_risk_pct == pytest.approx(0.75)


def test_stale_seven_percent_runtime_risk_is_clamped(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("PAPER_MODE", "false")
    monkeypatch.setenv("PAPER__ENABLED", "false")
    monkeypatch.setenv("SHADOW_MODE", "false")
    monkeypatch.setenv("RISK__PER_TRADE_RISK_PCT", "7.0")
    monkeypatch.setenv("RISK_PER_TRADE_PCT", "7.0")
    monkeypatch.delenv("RISK_PER_TRADE_HARD_CAP_PCT", raising=False)

    settings = _build_risk_settings()

    assert settings.per_trade_risk_pct == pytest.approx(0.75)


def test_risk_below_hard_cap_is_preserved(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("PAPER_MODE", "false")
    monkeypatch.setenv("PAPER__ENABLED", "false")
    monkeypatch.setenv("SHADOW_MODE", "false")
    monkeypatch.setenv("RISK__PER_TRADE_RISK_PCT", "0.50")
    monkeypatch.setenv("RISK_PER_TRADE_HARD_CAP_PCT", "0.75")

    settings = _build_risk_settings()

    assert settings.per_trade_risk_pct == pytest.approx(0.50)


def test_explicit_larger_hard_cap_allows_deliberate_policy_change(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("PAPER_MODE", "false")
    monkeypatch.setenv("PAPER__ENABLED", "false")
    monkeypatch.setenv("SHADOW_MODE", "false")
    monkeypatch.setenv("RISK__PER_TRADE_RISK_PCT", "0.80")
    monkeypatch.setenv("RISK_PER_TRADE_HARD_CAP_PCT", "1.00")

    settings = _build_risk_settings()

    assert settings.per_trade_risk_pct == pytest.approx(0.80)


def test_concurrent_same_symbol_distinct_signal_has_one_broker_submission(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    calls: list[str] = []
    first_at_broker = threading.Event()
    release_first = threading.Event()

    class _Broker:
        def place_order(self, **_kwargs):
            order_id = f"ENTRY-{len(calls) + 1}"
            calls.append(order_id)
            if len(calls) == 1:
                first_at_broker.set()
                assert release_first.wait(timeout=3.0)
            return {"order_id": order_id}

        def get_orders(self):
            return []

        def get_positions(self):
            return []

    class _Positions:
        def has_open_position(self, _symbol):
            return False

        def get_open_positions(self):
            return []

    manager = OrderManager(_Broker(), _Positions(), object())
    monkeypatch.setattr(manager, "_lot_size_for_symbol", lambda _symbol: 65)
    brackets = BracketManager(order_manager=manager)
    manager.set_bracket_manager(brackets)
    first_result: list[str | None] = []

    def _first_entry() -> None:
        first_result.append(
            manager.place_order(
                symbol=CE,
                side="BUY",
                quantity=65,
                order_type=OrderType.LIMIT,
                price=90.0,
                stop_loss=84.0,
                take_profit=104.0,
                intent="ENTRY",
                check_risk=False,
                signal_id="setup-a",
            )
        )

    thread = threading.Thread(target=_first_entry)
    try:
        thread.start()
        assert first_at_broker.wait(timeout=3.0)

        second = manager.place_order(
            symbol=CE,
            side="BUY",
            quantity=65,
            order_type=OrderType.LIMIT,
            price=90.0,
            stop_loss=84.0,
            take_profit=104.0,
            intent="ENTRY",
            check_risk=False,
            signal_id="setup-b",
        )

        release_first.set()
        thread.join(timeout=3.0)

        assert second is None
        assert first_result == ["ENTRY-1"]
        assert calls == ["ENTRY-1"]
    finally:
        release_first.set()
        thread.join(timeout=3.0)
        brackets._running = False

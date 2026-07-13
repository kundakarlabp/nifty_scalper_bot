from __future__ import annotations

import time
from types import SimpleNamespace

from nifty_scalper_bot.core import strategy_live_safety as guard
from nifty_scalper_bot.core.strategy_manager import StrategyManager
from nifty_scalper_bot.strategies.signal_generator import Signal


class _Engine:
    def __init__(
        self, bars: int, *, latest_age_s: float = 1.0, indicators: dict | None = None
    ) -> None:
        self.bars = bars
        self.latest_age_s = latest_age_s
        self.indicators = dict(indicators or {})

    def get_history(self, _symbol: str):
        ts = time.time() - self.latest_age_s
        return [
            {"timestamp": ts, "open": 1, "high": 1, "low": 1, "close": 1}
            for _ in range(self.bars)
        ]

    def get_indicators(self, _symbol: str, _names):
        return dict(self.indicators)


class _Positions:
    def get_position(self, _symbol: str):
        return None


def _live_env(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    monkeypatch.setenv("PAPER_MODE", "false")
    monkeypatch.setenv("PAPER__ENABLED", "false")
    monkeypatch.setenv("SHADOW_MODE", "false")


def _signal(**metadata) -> Signal:
    return Signal(
        action="BUY",
        symbol="NFO:NIFTY2662324050CE",
        quantity=1,
        confidence=0.9,
        reason="test",
        stop_loss=90.0,
        take_profit=120.0,
        metadata=dict(metadata),
    )


def test_live_strategy_manager_fails_closed_on_cold_history(monkeypatch) -> None:
    _live_env(monkeypatch)
    monkeypatch.setenv("OPTION_EVAL_MIN_BARS", "10")
    manager = StrategyManager([], _Engine(0), _Positions())
    manager._required_candles = 10

    result = manager.generate_signal(
        "NFO:NIFTY2662324050CE", 100.0, trace_id="cold-history"
    )

    assert result is None
    decision = manager.get_last_no_signal_decision("NFO:NIFTY2662324050CE")
    assert decision is not None
    assert decision.reason == "live_indicators_not_ready"
    assert decision.category == "live_safety"


def test_live_strategy_manager_fails_closed_when_hub_not_ready(monkeypatch) -> None:
    _live_env(monkeypatch)
    monkeypatch.setenv("OPTION_EVAL_MIN_BARS", "2")
    manager = StrategyManager([], _Engine(5), _Positions())
    manager._required_candles = 2
    manager._data_hub = SimpleNamespace(indicators_ready=False)

    result = manager.generate_signal(
        "NFO:NIFTY2662324050CE", 100.0, trace_id="hub-not-ready"
    )

    assert result is None
    decision = manager.get_last_no_signal_decision("NFO:NIFTY2662324050CE")
    assert decision is not None
    assert decision.reason == "live_hub_indicators_not_ready"


def test_live_approved_signal_must_pass_final_filter(monkeypatch) -> None:
    _live_env(monkeypatch)
    manager = SimpleNamespace(
        _filter_signal=lambda _signal: False,
        _orchestrator=None,
        _position_manager=None,
        _last_no_signal_decision_by_symbol={},
    )

    result = guard._final_filter(
        manager, _signal(is_approved=True, timestamp="2026-07-05T09:20:00"), "approved"
    )

    assert result is None
    decision = manager._last_no_signal_decision_by_symbol["NFO:NIFTY2662324050CE"]
    assert decision.reason == "live_signal_final_filter_block"


def test_live_signal_identity_is_generated_and_enriched() -> None:
    missing = _signal(strategy="SMC")
    present = _signal(strategy="SMC", timestamp="2026-07-05T09:20:00")

    assert guard._has_identity(missing) is False
    assert guard._has_identity(present) is True
    enriched_missing = guard._add_identity(missing)
    assert guard._has_identity(enriched_missing) is True
    assert (
        enriched_missing.metadata["deterministic_signal_id"] == missing.deterministic_id
    )
    assert enriched_missing.metadata["signal_id"] == missing.deterministic_id
    assert enriched_missing.metadata["idempotency_key"] == missing.deterministic_id
    enriched_present = guard._add_identity(present)
    assert (
        enriched_present.metadata["deterministic_signal_id"] == present.deterministic_id
    )


class _TickMDM:
    def __init__(self, age: float | None) -> None:
        self.age = age
        self.recovery_requests: list[tuple[str, str]] = []

    def time_since_last_live_ws_tick(self, symbol: str):
        return self.age

    def request_fallback_refresh(self, symbol: str, *, reason: str) -> bool:
        self.recovery_requests.append((symbol, reason))
        return True


class _CallingStrategy:
    name = "SMC"
    config = {}

    def __init__(self) -> None:
        self.calls = 0

    def get_required_indicators(self):
        return []

    def generate_signal(self, symbol, indicators, current_price, position=None):
        self.calls += 1
        return None


def _manager_with_strategy(
    engine: _Engine, strategy: _CallingStrategy | None = None
) -> tuple[StrategyManager, _CallingStrategy]:
    strategy = strategy or _CallingStrategy()
    manager = StrategyManager([strategy], engine, _Positions())
    manager._required_candles = 2
    return manager, strategy


def test_live_strategy_manager_fails_closed_on_stale_latest_bar(monkeypatch) -> None:
    _live_env(monkeypatch)
    monkeypatch.setenv("STRATEGY_LATEST_BAR_MAX_AGE_SECONDS", "10")
    manager, strategy = _manager_with_strategy(_Engine(5, latest_age_s=60.0))

    result = manager.generate_signal(
        "NFO:NIFTY2662324050CE", 100.0, trace_id="stale-bar"
    )

    assert result is None
    assert strategy.calls == 0
    assert (
        manager.get_last_no_signal_decision("NFO:NIFTY2662324050CE").reason
        == "live_latest_closed_bar_stale"
    )


def test_live_strategy_manager_blocks_stale_option_tick_and_requests_recovery(
    monkeypatch,
) -> None:
    _live_env(monkeypatch)
    monkeypatch.setenv("STRATEGY_LIVE_TICK_MAX_AGE_SECONDS", "5")
    manager, strategy = _manager_with_strategy(_Engine(5))
    mdm = _TickMDM(age=30.0)
    manager._market_data_manager = mdm

    result = manager.generate_signal(
        "NFO:NIFTY2662324050CE", 100.0, trace_id="stale-tick"
    )

    assert result is None
    assert strategy.calls == 0
    assert mdm.recovery_requests == [
        ("NFO:NIFTY2662324050CE", "strategy_live_option_tick_stale")
    ]
    assert (
        manager.get_last_no_signal_decision("NFO:NIFTY2662324050CE").reason
        == "live_option_tick_stale"
    )


def test_live_strategy_manager_fails_closed_on_stale_underlying_context(
    monkeypatch,
) -> None:
    _live_env(monkeypatch)
    engine = _Engine(5, indicators={"context_fresh": False})
    manager, strategy = _manager_with_strategy(engine)

    result = manager.generate_signal(
        "NFO:NIFTY2662324050CE", 100.0, trace_id="stale-context"
    )

    assert result is None
    assert strategy.calls == 0
    assert (
        manager.get_last_no_signal_decision("NFO:NIFTY2662324050CE").reason
        == "live_underlying_context_stale"
    )


def test_live_strategy_manager_fails_closed_on_selected_contract_mismatch(
    monkeypatch,
) -> None:
    _live_env(monkeypatch)
    engine = _Engine(
        5,
        indicators={
            "selected_ce": "NFO:NIFTY2662324050CE",
            "selected_pe": "NFO:NIFTY2662324050PE",
        },
    )
    manager, strategy = _manager_with_strategy(engine)

    result = manager.generate_signal(
        "NFO:NIFTY2662324100CE", 100.0, trace_id="contract-mismatch"
    )

    assert result is None
    assert strategy.calls == 0
    assert (
        manager.get_last_no_signal_decision("NFO:NIFTY2662324100CE").reason
        == "live_selected_contract_mismatch"
    )


def test_live_strategy_manager_invokes_strategy_when_canonical_freshness_gate_passes(
    monkeypatch,
) -> None:
    _live_env(monkeypatch)
    engine = _Engine(
        5, indicators={"context_fresh": True, "selected_ce": "NFO:NIFTY2662324050CE"}
    )
    manager, strategy = _manager_with_strategy(engine)
    manager._market_data_manager = _TickMDM(age=1.0)

    result = manager.generate_signal(
        "NFO:NIFTY2662324050CE", 100.0, trace_id="fresh-pass"
    )

    assert result is None
    assert strategy.calls == 1
    decision = manager.get_last_no_signal_decision("NFO:NIFTY2662324050CE")
    assert decision is None or decision.blocked_at != "strategy_live_safety"

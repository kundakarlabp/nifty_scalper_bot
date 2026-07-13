from __future__ import annotations

import time
from types import SimpleNamespace

from nifty_scalper_bot.core import strategy_live_safety as guard
from nifty_scalper_bot.core.strategy_manager import StrategyManager
from nifty_scalper_bot.strategies.signal_generator import Signal


class _Engine:
    def __init__(
        self,
        bars: int,
        *,
        latest_age_s: float = 1.0,
        indicators: dict | None = None,
        raises: bool = False,
        latest_extra: dict | None = None,
        rows: list[dict] | None = None,
    ) -> None:
        self.bars = bars
        self.latest_age_s = latest_age_s
        self.indicators = dict(indicators or {})
        self.raises = raises
        self.latest_extra = dict(latest_extra or {})
        self.rows = [dict(row) for row in rows] if rows is not None else None
        self.indicator_calls = 0

    def get_history(self, _symbol: str):
        if self.raises:
            raise RuntimeError("history unavailable")
        if self.rows is not None:
            return [dict(row) for row in self.rows]
        ts = time.time() - self.latest_age_s
        row = {"timestamp": ts, "open": 1, "high": 1, "low": 1, "close": 1}
        row.update(self.latest_extra)
        return [dict(row) for _ in range(self.bars)]

    def get_indicators(self, _symbol: str, _names):
        self.indicator_calls += 1
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


def _shadow_env(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("ENABLE_LIVE", "false")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "false")
    monkeypatch.setenv("SHADOW_MODE", "true")


def _set_now(monkeypatch, epoch: float) -> None:
    monkeypatch.setattr(guard.time, "time", lambda: epoch)


def _bars(*timestamps: float, extra: dict | None = None) -> list[dict]:
    extra = dict(extra or {})
    return [
        dict({"timestamp": ts, "open": 1, "high": 1, "low": 1, "close": 1}, **extra)
        for ts in timestamps
    ]


def _history_at(
    timestamp: float, *, count: int = 5, extra: dict | None = None
) -> list[dict]:
    return _bars(*([timestamp] * count), extra=extra)


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
    def __init__(self, age: float | None, *, basket: dict | None = None) -> None:
        self.age = age
        self.basket = basket or {
            "selected_ce": "NFO:NIFTY2662324050CE",
            "selected_pe": "NFO:NIFTY2662324050PE",
            "basket_version": "v1",
        }
        self.recovery_requests: list[tuple[str, str]] = []

    def time_since_last_live_ws_tick(self, symbol: str):
        return self.age

    def request_fallback_refresh(self, symbol: str, *, reason: str) -> bool:
        self.recovery_requests.append((symbol, reason))
        return True

    def get_active_contract_basket(self):
        return self.basket


class _NoFreshnessMDM:
    def get_active_contract_basket(self):
        return {
            "selected_ce": "NFO:NIFTY2662324050CE",
            "selected_pe": "NFO:NIFTY2662324050PE",
        }


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


def _fresh_context(*, spot_age: float = 1.0, futures_age: float = 1.0) -> dict:
    now = time.time()
    return {
        "spot_context": {"timestamp": now - spot_age},
        "futures_context": {"timestamp": now - futures_age},
    }


def _manager_with_strategy(
    engine: _Engine, strategy: _CallingStrategy | None = None
) -> tuple[StrategyManager, _CallingStrategy]:
    strategy = strategy or _CallingStrategy()
    if engine.rows is None and not engine.raises:
        now = guard.time.time()
        engine.rows = _history_at((now // 60.0) * 60.0 - 60.0)
    manager = StrategyManager([strategy], engine, _Positions())
    manager._required_candles = 2
    manager._bar_interval_seconds = 60.0
    manager._market_data_manager = _TickMDM(age=1.0)
    manager._latest_context_snapshots = _fresh_context()
    manager._active_basket_version = "v1"
    return manager, strategy


def test_live_strategy_manager_accepts_expected_closed_bucket_early_next_minute(
    monkeypatch,
) -> None:
    _live_env(monkeypatch)
    now = 11 * 3600 + 31 * 60 + 5
    _set_now(monkeypatch, float(now))
    manager, strategy = _manager_with_strategy(
        _Engine(5, rows=_history_at(11 * 3600 + 30 * 60))
    )

    result = manager.generate_signal(
        "NFO:NIFTY2662324050CE", 100.0, trace_id="fresh-bar-early"
    )

    assert result is None
    assert strategy.calls == 1


def test_live_strategy_manager_accepts_expected_closed_bucket_late_next_minute(
    monkeypatch,
) -> None:
    _live_env(monkeypatch)
    now = 11 * 3600 + 31 * 60 + 50
    _set_now(monkeypatch, float(now))
    manager, strategy = _manager_with_strategy(
        _Engine(5, rows=_history_at(11 * 3600 + 30 * 60))
    )

    result = manager.generate_signal(
        "NFO:NIFTY2662324050CE", 100.0, trace_id="fresh-bar-late"
    )

    assert result is None
    assert strategy.calls == 1


def test_live_strategy_manager_fails_closed_on_one_interval_old_bar(
    monkeypatch,
) -> None:
    _live_env(monkeypatch)
    now = 11 * 3600 + 32 * 60 + 5
    _set_now(monkeypatch, float(now))
    manager, strategy = _manager_with_strategy(
        _Engine(5, rows=_history_at(11 * 3600 + 30 * 60))
    )

    result = manager.generate_signal(
        "NFO:NIFTY2662324050CE", 100.0, trace_id="stale-bar"
    )

    assert result is None
    assert strategy.calls == 0
    assert (
        manager.get_last_no_signal_decision("NFO:NIFTY2662324050CE").reason
        == "live_latest_closed_bar_stale"
    )


def test_live_strategy_manager_ignores_current_forming_bucket_as_closed(
    monkeypatch,
) -> None:
    _live_env(monkeypatch)
    now = 11 * 3600 + 31 * 60 + 5
    _set_now(monkeypatch, float(now))
    manager, strategy = _manager_with_strategy(
        _Engine(5, rows=_history_at(11 * 3600 + 31 * 60))
    )

    result = manager.generate_signal("NFO:NIFTY2662324050CE", 100.0)

    assert result is None
    assert strategy.calls == 0
    assert (
        manager.get_last_no_signal_decision("NFO:NIFTY2662324050CE").reason
        == "live_latest_closed_bar_open"
    )


def test_live_strategy_manager_blocks_stale_option_tick_and_requests_recovery(
    monkeypatch,
) -> None:
    _live_env(monkeypatch)
    monkeypatch.setenv("HYDRATION_LIVE_TICK_MAX_AGE_SECONDS", "5")
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
    engine = _Engine(5)
    manager, strategy = _manager_with_strategy(engine)
    manager._latest_context_snapshots = _fresh_context(spot_age=1.0, futures_age=300.0)

    result = manager.generate_signal(
        "NFO:NIFTY2662324050CE", 100.0, trace_id="stale-context"
    )

    assert result is None
    assert strategy.calls == 0
    assert (
        manager.get_last_no_signal_decision("NFO:NIFTY2662324050CE").reason
        == "live_futures_context_stale"
    )


def test_live_strategy_manager_fails_closed_on_selected_contract_mismatch(
    monkeypatch,
) -> None:
    _live_env(monkeypatch)
    engine = _Engine(5)
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
    engine = _Engine(5)
    manager, strategy = _manager_with_strategy(engine)

    result = manager.generate_signal(
        "NFO:NIFTY2662324050CE", 100.0, trace_id="fresh-pass"
    )

    assert result is None
    assert strategy.calls == 1
    decision = manager.get_last_no_signal_decision("NFO:NIFTY2662324050CE")
    assert decision is None or decision.blocked_at != "strategy_live_safety"


def test_live_strategy_manager_fails_closed_when_mdm_missing(monkeypatch) -> None:
    _live_env(monkeypatch)
    manager, strategy = _manager_with_strategy(_Engine(5))
    manager._market_data_manager = None
    manager._data_hub = None

    result = manager.generate_signal("NFO:NIFTY2662324050CE", 100.0)

    assert result is None
    assert strategy.calls == 0
    assert (
        manager.get_last_no_signal_decision("NFO:NIFTY2662324050CE").reason
        == "live_market_data_manager_missing"
    )


def test_live_strategy_manager_fails_closed_when_tick_freshness_api_missing(
    monkeypatch,
) -> None:
    _live_env(monkeypatch)
    manager, strategy = _manager_with_strategy(_Engine(5))
    manager._market_data_manager = _NoFreshnessMDM()

    result = manager.generate_signal("NFO:NIFTY2662324050CE", 100.0)

    assert result is None
    assert strategy.calls == 0
    assert (
        manager.get_last_no_signal_decision("NFO:NIFTY2662324050CE").reason
        == "live_option_tick_freshness_unavailable"
    )


def test_live_strategy_manager_fails_closed_when_first_ws_tick_missing(
    monkeypatch,
) -> None:
    _live_env(monkeypatch)
    manager, strategy = _manager_with_strategy(_Engine(5))
    manager._market_data_manager = _TickMDM(age=None)

    result = manager.generate_signal("NFO:NIFTY2662324050CE", 100.0)

    assert result is None
    assert strategy.calls == 0
    assert (
        manager.get_last_no_signal_decision("NFO:NIFTY2662324050CE").reason
        == "live_option_tick_missing"
    )


def test_live_strategy_manager_history_exception_fails_closed_without_crash(
    monkeypatch,
) -> None:
    _live_env(monkeypatch)
    manager, strategy = _manager_with_strategy(_Engine(5, raises=True))

    result = manager.generate_signal("NFO:NIFTY2662324050CE", 100.0)

    assert result is None
    assert strategy.calls == 0
    assert (
        manager.get_last_no_signal_decision("NFO:NIFTY2662324050CE").reason
        == "live_indicators_not_ready"
    )


def test_live_strategy_manager_blocks_when_spot_stale_even_if_future_fresh(
    monkeypatch,
) -> None:
    _live_env(monkeypatch)
    manager, strategy = _manager_with_strategy(_Engine(5))
    manager._latest_context_snapshots = _fresh_context(spot_age=300.0, futures_age=1.0)

    result = manager.generate_signal("NFO:NIFTY2662324050CE", 100.0)

    assert result is None
    assert strategy.calls == 0
    assert (
        manager.get_last_no_signal_decision("NFO:NIFTY2662324050CE").reason
        == "live_spot_context_stale"
    )


def test_live_strategy_manager_blocks_when_context_freshness_unknown(
    monkeypatch,
) -> None:
    _live_env(monkeypatch)
    manager, strategy = _manager_with_strategy(_Engine(5))
    manager._latest_context_snapshots = {}

    result = manager.generate_signal("NFO:NIFTY2662324050CE", 100.0)

    assert result is None
    assert strategy.calls == 0
    assert (
        manager.get_last_no_signal_decision("NFO:NIFTY2662324050CE").reason
        == "live_spot_context_missing"
    )


def test_live_strategy_manager_blocks_when_active_basket_missing(monkeypatch) -> None:
    _live_env(monkeypatch)
    manager, strategy = _manager_with_strategy(_Engine(5))
    manager._market_data_manager = _TickMDM(age=1.0, basket=None)
    manager._market_data_manager.basket = None

    result = manager.generate_signal("NFO:NIFTY2662324050CE", 100.0)

    assert result is None
    assert strategy.calls == 0
    assert (
        manager.get_last_no_signal_decision("NFO:NIFTY2662324050CE").reason
        == "live_active_basket_missing"
    )


def test_live_strategy_manager_blocks_canonical_basket_version_mismatch(
    monkeypatch,
) -> None:
    _live_env(monkeypatch)
    manager, strategy = _manager_with_strategy(_Engine(5))
    manager._market_data_manager = _TickMDM(
        age=1.0,
        basket={
            "selected_ce": "NFO:NIFTY2662324050CE",
            "selected_pe": "NFO:NIFTY2662324050PE",
            "basket_version": "v2",
        },
    )

    result = manager.generate_signal("NFO:NIFTY2662324050CE", 100.0)

    assert result is None
    assert strategy.calls == 0
    assert (
        manager.get_last_no_signal_decision("NFO:NIFTY2662324050CE").reason
        == "live_selected_contract_version_stale"
    )


def test_live_strategy_manager_does_not_prime_indicators_before_dispatch(
    monkeypatch,
) -> None:
    _live_env(monkeypatch)
    engine = _Engine(5)
    manager, strategy = _manager_with_strategy(engine)

    result = manager.generate_signal("NFO:NIFTY2662324050CE", 100.0)

    assert result is None
    assert strategy.calls == 1
    assert engine.indicator_calls == 1


def test_live_strategy_manager_blocks_latest_open_candle(monkeypatch) -> None:
    _live_env(monkeypatch)
    now = 11 * 3600 + 31 * 60 + 5
    _set_now(monkeypatch, float(now))
    manager, strategy = _manager_with_strategy(
        _Engine(5, rows=_history_at(11 * 3600 + 30 * 60, extra={"is_closed": False}))
    )

    result = manager.generate_signal("NFO:NIFTY2662324050CE", 100.0)

    assert result is None
    assert strategy.calls == 0
    assert (
        manager.get_last_no_signal_decision("NFO:NIFTY2662324050CE").reason
        == "live_latest_closed_bar_open"
    )


def test_live_strategy_manager_blocks_future_bar_timestamp(monkeypatch) -> None:
    _live_env(monkeypatch)
    now = 11 * 3600 + 31 * 60 + 5
    _set_now(monkeypatch, float(now))
    manager, strategy = _manager_with_strategy(_Engine(5, rows=_history_at(now + 30)))

    result = manager.generate_signal("NFO:NIFTY2662324050CE", 100.0)

    assert result is None
    assert strategy.calls == 0
    assert (
        manager.get_last_no_signal_decision("NFO:NIFTY2662324050CE").reason
        == "live_latest_closed_bar_future_timestamp"
    )


def test_live_strategy_manager_blocks_future_spot_context_timestamp(
    monkeypatch,
) -> None:
    _live_env(monkeypatch)
    now = 11 * 3600 + 31 * 60 + 5
    _set_now(monkeypatch, float(now))
    manager, strategy = _manager_with_strategy(
        _Engine(5, rows=_history_at(11 * 3600 + 30 * 60))
    )
    manager._latest_context_snapshots = {
        "spot_context": {"timestamp": now + 30},
        "futures_context": {"timestamp": now - 1},
    }

    result = manager.generate_signal("NFO:NIFTY2662324050CE", 100.0)

    assert result is None
    assert strategy.calls == 0
    assert (
        manager.get_last_no_signal_decision("NFO:NIFTY2662324050CE").reason
        == "live_spot_context_future_timestamp"
    )


def test_live_strategy_manager_blocks_future_futures_context_timestamp(
    monkeypatch,
) -> None:
    _live_env(monkeypatch)
    now = 11 * 3600 + 31 * 60 + 5
    _set_now(monkeypatch, float(now))
    manager, strategy = _manager_with_strategy(
        _Engine(5, rows=_history_at(11 * 3600 + 30 * 60))
    )
    manager._latest_context_snapshots = {
        "spot_context": {"timestamp": now - 1},
        "futures_context": {"timestamp": now + 30},
    }

    result = manager.generate_signal("NFO:NIFTY2662324050CE", 100.0)

    assert result is None
    assert strategy.calls == 0
    assert (
        manager.get_last_no_signal_decision("NFO:NIFTY2662324050CE").reason
        == "live_futures_context_future_timestamp"
    )


def test_live_strategy_manager_production_wired_fresh_runtime_invokes_once(
    monkeypatch,
) -> None:
    _shadow_env(monkeypatch)
    now = 11 * 3600 + 31 * 60 + 5
    _set_now(monkeypatch, float(now))
    strategy = _CallingStrategy()
    engine = _Engine(
        5,
        rows=_history_at(11 * 3600 + 30 * 60),
        indicators={"ltp": 100.0, "close": 100.0},
    )
    if engine.rows is None and not engine.raises:
        now = guard.time.time()
        engine.rows = _history_at((now // 60.0) * 60.0 - 60.0)
    manager = StrategyManager([strategy], engine, _Positions())
    manager._required_candles = 2
    manager._bar_interval_seconds = 60.0
    mdm = _TickMDM(age=1.0)
    manager._market_data_manager = mdm
    manager._active_basket_version = "v1"

    manager.generate_signal("NSE:NIFTY", 100.0, trace_id="publish-spot")
    manager.generate_signal("NFO:NIFTY26JUN25000FUT", 100.0, trace_id="publish-futures")

    _live_env(monkeypatch)
    result = manager.generate_signal(
        "NFO:NIFTY2662324050CE", 100.0, trace_id="live-option"
    )

    assert result is None
    assert strategy.calls == 1
    assert mdm.get_active_contract_basket()["selected_ce"] == "NFO:NIFTY2662324050CE"
    assert set(manager._latest_context_snapshots) >= {"spot_context", "futures_context"}

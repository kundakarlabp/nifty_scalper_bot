from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from nifty_scalper_bot.strategies.bar_builder import OneMinuteBar
from nifty_scalper_bot.strategies.runner import (
    RunnerState,
    SymbolState,
    StrategyRunner,
    StrategyRunnerConfig,
)


def _runner() -> StrategyRunner:
    return StrategyRunner(
        market_data_manager=MagicMock(),
        indicator_engine=MagicMock(),
        strategy_manager=MagicMock(),
        risk_manager=MagicMock(),
        order_manager=MagicMock(),
        position_manager=MagicMock(),
        config=StrategyRunnerConfig(max_trade_history=5),
        data_hub=MagicMock(),
    )


def test_validate_symbol_universe_rejects_missing_option_symbols() -> None:
    runner = _runner()
    runner._active_symbols = {"NSE:NIFTY"}

    assert runner._validate_symbol_universe() is False


def test_validate_symbol_universe_rejects_index_count_mismatch() -> None:
    runner = _runner()
    runner._active_symbols = {
        "NSE:NIFTY",
        "NSE:NIFTY BANK",
        "NIFTY25JAN25000CE",
        "NIFTY25JAN25000PE",
    }

    assert runner._validate_symbol_universe() is False


def test_validate_symbol_universe_accepts_index_and_options() -> None:
    runner = _runner()
    runner._active_symbols = {
        "NSE:NIFTY",
        "NIFTY25JAN25000CE",
        "NIFTY25JAN25000PE",
    }

    assert runner._validate_symbol_universe() is True


def test_mark_symbol_unready_sets_state_flags() -> None:
    runner = _runner()
    symbol = "NIFTY25JAN25000CE"
    runner.add_symbol(symbol)

    runner._mark_symbol_unready(symbol, "insufficient_history", low_confidence=True)

    state = runner._symbol_state[symbol]
    assert runner._history_ready_by_symbol[symbol] is False
    assert state.strategy_data["unready_reason"] == "insufficient_history"
    assert state.strategy_data["low_confidence"] is True


def test_validate_symbol_for_cycle_debugs_and_skips_when_symbol_not_active(
    caplog,
) -> None:
    runner = _runner()
    symbol = "NIFTY25JAN25000CE"
    runner._active_symbols = {"NIFTY25JAN25000PE"}

    with caplog.at_level("DEBUG", logger=runner._logger.name):
        ok = runner._validate_symbol_for_cycle(symbol)

    assert ok is False
    assert any(
        getattr(record, "event", "") == "symbol_outside_active_universe"
        for record in caplog.records
    )


def test_validate_symbol_for_cycle_skips_quarantined_symbol(caplog) -> None:
    runner = _runner()
    symbol = "NIFTY25JAN25000CE"
    runner._active_symbols = {symbol}
    runner._quarantined_symbols.add(symbol)

    with caplog.at_level("WARNING", logger=runner._logger.name):
        ok = runner._validate_symbol_for_cycle(symbol)

    assert ok is False
    assert any(
        getattr(record, "event", "") == "symbol_quarantined_skip"
        for record in caplog.records
    )


def test_set_data_freshness_backoff_rate_limit_marks_symbol_not_global_pause(
    caplog,
) -> None:
    runner = _runner()
    symbol = "NIFTY25JAN25000CE"
    runner.add_symbol(symbol)

    with caplog.at_level("WARNING", logger=runner._logger.name):
        runner.set_data_freshness_backoff(
            2.0,
            detail_code="rate_limit_429",
            symbol=symbol,
        )

    assert symbol in runner._rate_limit_backoff_until_by_symbol
    assert runner._trading_paused is False
    assert any(
        getattr(record, "event", "") == "rate_limit_breach" for record in caplog.records
    )


def test_warn_symbol_gate_emits_structured_warning(caplog) -> None:
    runner = _runner()

    with caplog.at_level("WARNING", logger=runner._logger.name):
        runner._warn_symbol_gate(
            "indicator_invalid",
            "NIFTY25JAN25000CE",
            "Indicators are invalid",
            reason="min_bars_not_ready",
        )

    match = [
        record
        for record in caplog.records
        if getattr(record, "event", "") == "indicator_invalid"
    ]
    assert match
    record = match[0]
    assert getattr(record, "level", "") == "WARNING"
    assert getattr(record, "symbol", "") == "NIFTY25JAN25000CE"


def test_warn_symbol_gate_sanitizes_reserved_logrecord_context_keys(caplog) -> None:
    runner = _runner()

    with caplog.at_level("WARNING", logger=runner._logger.name):
        runner._warn_symbol_gate(
            "indicator_invalid",
            "NIFTY25JAN25000CE",
            "Indicators are invalid",
            reason="min_bars_not_ready",
            message="reserved_message",
        )

    match = [
        record
        for record in caplog.records
        if getattr(record, "event", "") == "indicator_invalid"
    ]
    assert match
    record = match[0]
    assert getattr(record, "gate_message", "") == "Indicators are invalid"
    assert getattr(record, "context_message", "") == "reserved_message"


def test_on_tick_insufficient_history_skips_symbol_until_hydrated(
    caplog, monkeypatch
) -> None:
    runner = _runner()
    symbol = "NIFTY25JAN25000CE"
    runner.add_symbol(symbol)
    runner._startup_timestamp = 0.0
    runner._active_symbols = {symbol, "NSE:NIFTY"}
    runner._history_ready_by_symbol[symbol] = False
    monkeypatch.setattr(runner, "_is_market_open", lambda _now: True)

    with caplog.at_level("WARNING", logger=runner._logger.name):
        runner._on_tick(symbol, {"ltp": 101.0, "timestamp": 1_700_000_000})

    assert any(
        getattr(record, "event", "") == "symbol_hydrating" for record in caplog.records
    )
    assert runner._strategy_manager.generate_signal.call_count == 0


def test_on_tick_hydration_barrier_blocks_execution(caplog, monkeypatch) -> None:
    runner = _runner()
    symbol = "NIFTY25JAN25000CE"
    runner.add_symbol(symbol)
    runner._startup_timestamp = 0.0
    runner._active_symbols = {symbol, "NSE:NIFTY"}
    runner._history_ready_by_symbol[symbol] = True
    runner._required_candles = 1
    runner._runner_state = RunnerState.EXECUTION_ENABLED
    runner._hydration_complete = False
    monkeypatch.setattr(runner, "_is_market_open", lambda _now: True)

    with caplog.at_level("WARNING", logger=runner._logger.name):
        runner._on_tick(symbol, {"ltp": 101.0, "timestamp": 1_700_000_000})

    assert any(
        getattr(record, "event", "") == "execution_blocked_hydration"
        for record in caplog.records
    )
    assert runner._strategy_manager.generate_signal.call_count == 0


def test_on_tick_missing_finalized_bar_warns_and_skips_symbol(
    caplog, monkeypatch
) -> None:
    runner = _runner()
    symbol = "NIFTY25JAN25000CE"
    runner.add_symbol(symbol)
    runner._startup_timestamp = 0.0
    runner._active_symbols = {symbol, "NSE:NIFTY"}
    runner._history_ready_by_symbol[symbol] = True
    runner._required_candles = 1
    runner._symbol_state[symbol].vwap = 100.0
    monkeypatch.setattr(runner, "_is_market_open", lambda _now: True)
    runner._indicator_engine.ensure_min_bars.return_value = True
    runner._indicator_engine.has_min_bars.return_value = True

    with caplog.at_level("WARNING", logger=runner._logger.name):
        runner._on_tick(symbol, {"ltp": 101.0, "timestamp": 1_700_000_000})

    assert any(
        getattr(record, "event", "") == "bar_not_finalized" for record in caplog.records
    )
    assert runner._strategy_manager.generate_signal.call_count == 0


def test_on_tick_invalid_vwap_skips_symbol(caplog, monkeypatch) -> None:
    runner = _runner()
    symbol = "NIFTY25JAN25000CE"
    runner.add_symbol(symbol)
    runner._startup_timestamp = 0.0
    runner._active_symbols = {symbol, "NSE:NIFTY"}
    runner._history_ready_by_symbol[symbol] = True
    runner._required_candles = 1
    monkeypatch.setattr(runner, "_is_market_open", lambda _now: True)

    with caplog.at_level("WARNING", logger=runner._logger.name):
        runner._on_tick(symbol, {"ltp": 101.0, "timestamp": 1_700_000_000})

    assert any(
        getattr(record, "event", "") == "symbol_hydrating" for record in caplog.records
    )
    assert runner._strategy_manager.generate_signal.call_count == 0


def test_rate_limit_backoff_skips_only_target_symbol(caplog, monkeypatch) -> None:
    runner = _runner()
    blocked_symbol = "NIFTY25JAN25000CE"
    ready_symbol = "NIFTY25JAN25000PE"
    runner.add_symbol(blocked_symbol)
    runner.add_symbol(ready_symbol)
    runner._startup_timestamp = 0.0
    runner._active_symbols = {blocked_symbol, ready_symbol, "NSE:NIFTY"}
    runner._required_candles = 1
    runner._history_ready_by_symbol[blocked_symbol] = True
    runner._history_ready_by_symbol[ready_symbol] = True
    runner._symbol_state[ready_symbol].vwap = 100.0
    runner._last_bar_ts[ready_symbol] = datetime.now(timezone.utc)
    runner._indicator_engine.ensure_min_bars.return_value = True
    runner._indicator_engine.has_min_bars.return_value = True
    runner._strategy_manager.generate_signal.return_value = None
    monkeypatch.setattr(runner, "_is_market_open", lambda _now: True)

    runner.set_data_freshness_backoff(
        5.0,
        detail_code="rate_limit_429",
        symbol=blocked_symbol,
    )

    with caplog.at_level("WARNING", logger=runner._logger.name):
        runner._on_tick(blocked_symbol, {"ltp": 101.0, "timestamp": 1_700_000_000})
        runner._on_tick(ready_symbol, {"ltp": 101.0, "timestamp": 1_700_000_000})

    assert any(
        getattr(record, "event", "") == "rate_limit_breach" for record in caplog.records
    )
    assert runner._strategy_manager.generate_signal.call_count == 1


def test_update_symbol_hydration_requires_two_valid_bars() -> None:
    runner = _runner()
    symbol = "NIFTY25JAN25000CE"
    runner.add_symbol(symbol)
    runner._required_candles = 2

    state_one = runner.update_symbol_hydration(
        symbol,
        [1.0, 2.0],
        {symbol: {"vwap": 100.0, "cum_volume": 10}},
    )
    state_two = runner.update_symbol_hydration(
        symbol,
        [1.0, 2.0],
        {symbol: {"vwap": 101.0, "cum_volume": 20}},
    )

    assert state_one.name == "HYDRATING"
    assert state_two.name == "READY"


def test_ready_symbol_does_not_downgrade_without_session_reset() -> None:
    runner = _runner()
    symbol = "NIFTY25JAN25000CE"
    runner.add_symbol(symbol)
    runner._set_symbol_hydration_state(symbol, SymbolState.READY)

    state = runner.update_symbol_hydration(
        symbol,
        [1.0] * runner._required_candles,
        {symbol: {"vwap": 0.0, "cum_volume": 0}},
    )

    assert state.name == "READY"


def test_repair_candle_gap_creates_synthetic_when_history_missing() -> None:
    runner = _runner()
    symbol = "NIFTY25JAN25000CE"
    prev = OneMinuteBar(
        open=100.0,
        high=101.0,
        low=99.5,
        close=100.5,
        volume=10,
        start=datetime(2026, 1, 1, 9, 15, tzinfo=timezone.utc),
        end=datetime(2026, 1, 1, 9, 15, 59, tzinfo=timezone.utc),
    )
    incoming = OneMinuteBar(
        open=103.0,
        high=104.0,
        low=102.0,
        close=103.5,
        volume=8,
        start=datetime(2026, 1, 1, 9, 18, tzinfo=timezone.utc),
        end=datetime(2026, 1, 1, 9, 18, 59, tzinfo=timezone.utc),
    )

    repaired = runner._repair_candle_gap(symbol, prev, incoming)

    assert len(repaired) == 2
    assert repaired[0].start == prev.start + timedelta(minutes=1)
    assert repaired[0].open == pytest.approx(prev.close)
    assert repaired[0].volume == 0
    assert repaired[1].start == prev.start + timedelta(minutes=2)


def test_version_guard_blocks_stale_strategy_evaluation(monkeypatch) -> None:
    runner = _runner()
    symbol = "NIFTY25JAN25000CE"
    runner.add_symbol(symbol)
    runner._running = True
    runner._trading_paused = False
    runner._runner_state = runner._runner_state.EXECUTION_ENABLED
    runner._active_symbols = {symbol, "NSE:NIFTY"}
    runner._history_ready_by_symbol[symbol] = True
    runner._required_candles = 1
    runner._symbol_state[symbol].vwap = 100.0
    runner._candle_versions[symbol] = 1
    runner._last_strategy_versions[symbol] = 1
    monkeypatch.setattr(runner, "_is_market_open", lambda _now: True)

    runner._on_tick(symbol, {"ltp": 101.0, "timestamp": 1_700_000_000})

    assert runner._strategy_manager.generate_signal.call_count == 0


def test_composite_reports_emit_aggregate_logs(caplog) -> None:
    runner = _runner()
    runner._last_system_heartbeat_log = 0.0
    runner._last_strategy_status_log = 0.0
    runner._strategy_window_symbols = {"A", "B"}
    runner._strategy_window_signals = 3

    with caplog.at_level("INFO", logger=runner._logger.name):
        runner._emit_composite_reports()

    events = {getattr(record, "event", "") for record in caplog.records}
    assert "system_heartbeat" in events
    assert "strategy_status_report" in events

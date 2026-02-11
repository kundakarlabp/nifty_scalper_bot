from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

from nifty_scalper_bot.strategies.runner import StrategyRunner, StrategyRunnerConfig


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
    runner._active_symbols = {"NSE:NIFTY 50"}

    assert runner._validate_symbol_universe() is False


def test_validate_symbol_universe_rejects_index_count_mismatch() -> None:
    runner = _runner()
    runner._active_symbols = {
        "NSE:NIFTY 50",
        "NSE:NIFTY BANK",
        "NIFTY25JAN25000CE",
        "NIFTY25JAN25000PE",
    }

    assert runner._validate_symbol_universe() is False


def test_validate_symbol_universe_accepts_index_and_options() -> None:
    runner = _runner()
    runner._active_symbols = {
        "NSE:NIFTY 50",
        "NIFTY25JAN25000CE",
        "NIFTY25JAN25000PE",
    }

    assert runner._validate_symbol_universe() is True


def test_validate_symbol_universe_rejects_expected_count_deviation() -> None:
    runner = _runner()
    runner._active_symbols = {
        "NSE:NIFTY 50",
        "NIFTY25JAN25000CE",
        "NIFTY25JAN25000PE",
    }
    runner._frozen_symbol_universe = set(runner._active_symbols)
    runner._expected_symbol_count = 2

    assert runner._validate_symbol_universe() is False


def test_validate_symbol_universe_rejects_frozen_universe_deviation() -> None:
    runner = _runner()
    runner._active_symbols = {
        "NSE:NIFTY 50",
        "NIFTY25JAN25000CE",
        "NIFTY25JAN25000PE",
    }
    runner._frozen_symbol_universe = {
        "NSE:NIFTY 50",
        "NIFTY25JAN25000CE",
        "NIFTY25JAN25100PE",
    }
    runner._expected_symbol_count = 3

    assert runner._validate_symbol_universe() is False


def test_mark_symbol_unready_sets_state_flags() -> None:
    runner = _runner()
    symbol = "NIFTY25JAN25000CE"
    runner.add_symbol(symbol)

    runner._mark_symbol_unready(symbol, "insufficient_history", low_confidence=True)

    state = runner._symbol_state[symbol]
    assert runner._history_ready_by_symbol[symbol] is False
    assert state.strategy_data["unready_reason"] == "insufficient_history"
    assert state.strategy_data["low_confidence"] is True


def test_validate_symbol_for_cycle_warns_and_skips_on_universe_violation(
    caplog,
) -> None:
    runner = _runner()
    symbol = "NIFTY25JAN25000CE"
    runner._active_symbols = {symbol}
    runner._frozen_symbol_universe = {"NIFTY25JAN25000PE"}
    runner._expected_symbol_count = 1

    with caplog.at_level("WARNING", logger=runner._logger.name):
        ok = runner._validate_symbol_for_cycle(symbol)

    assert ok is False
    assert any(
        getattr(record, "event", "") == "universe_violation"
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


def test_on_tick_insufficient_history_warns_and_skips_symbol(
    caplog, monkeypatch
) -> None:
    runner = _runner()
    symbol = "NIFTY25JAN25000CE"
    runner.add_symbol(symbol)
    runner._startup_timestamp = 0.0
    runner._frozen_symbol_universe = {symbol, "NSE:NIFTY 50"}
    runner._active_symbols = {symbol, "NSE:NIFTY 50"}
    runner._expected_symbol_count = 2
    runner._history_ready_by_symbol[symbol] = False
    monkeypatch.setattr(runner, "_is_market_open", lambda _now: True)

    with caplog.at_level("WARNING", logger=runner._logger.name):
        runner._on_tick(symbol, {"ltp": 101.0, "timestamp": 1_700_000_000})

    assert any(
        getattr(record, "event", "") == "insufficient_history"
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
    runner._frozen_symbol_universe = {symbol, "NSE:NIFTY 50"}
    runner._active_symbols = {symbol, "NSE:NIFTY 50"}
    runner._expected_symbol_count = 2
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


def test_on_tick_invalid_vwap_warns_and_skips_symbol(caplog, monkeypatch) -> None:
    runner = _runner()
    symbol = "NIFTY25JAN25000CE"
    runner.add_symbol(symbol)
    runner._startup_timestamp = 0.0
    runner._frozen_symbol_universe = {symbol, "NSE:NIFTY 50"}
    runner._active_symbols = {symbol, "NSE:NIFTY 50"}
    runner._expected_symbol_count = 2
    runner._history_ready_by_symbol[symbol] = True
    runner._required_candles = 1
    monkeypatch.setattr(runner, "_is_market_open", lambda _now: True)

    with caplog.at_level("WARNING", logger=runner._logger.name):
        runner._on_tick(symbol, {"ltp": 101.0, "timestamp": 1_700_000_000})

    assert any(
        getattr(record, "event", "") == "vwap_invalid" for record in caplog.records
    )
    assert runner._strategy_manager.generate_signal.call_count == 0


def test_rate_limit_backoff_skips_only_target_symbol(caplog, monkeypatch) -> None:
    runner = _runner()
    blocked_symbol = "NIFTY25JAN25000CE"
    ready_symbol = "NIFTY25JAN25000PE"
    runner.add_symbol(blocked_symbol)
    runner.add_symbol(ready_symbol)
    runner._startup_timestamp = 0.0
    runner._frozen_symbol_universe = {blocked_symbol, ready_symbol, "NSE:NIFTY 50"}
    runner._active_symbols = {blocked_symbol, ready_symbol, "NSE:NIFTY 50"}
    runner._expected_symbol_count = 3
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

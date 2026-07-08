from __future__ import annotations

from pathlib import Path

from nifty_scalper_bot.strategies.runner import StrategyRunner


def test_option_reason_cooldown_key_is_side_aware() -> None:
    pe_key = StrategyRunner._reason_order_cooldown_key(
        underlying="NIFTY", option_side="PE", reason_key="OrderFlow"
    )
    ce_key = StrategyRunner._reason_order_cooldown_key(
        underlying="NIFTY", option_side="CE", reason_key="OrderFlow"
    )
    assert pe_key == "NIFTY:PE:OrderFlow"
    assert ce_key == "NIFTY:CE:OrderFlow"
    reason_cache = {pe_key: 1000.0}
    assert ce_key not in reason_cache
    assert pe_key in reason_cache


def test_same_option_reason_cooldown_key_still_blocks_same_side() -> None:
    first = StrategyRunner._reason_order_cooldown_key(
        underlying="NIFTY", option_side="PE", reason_key="OrderFlow"
    )
    second = StrategyRunner._reason_order_cooldown_key(
        underlying="NIFTY", option_side="PE", reason_key="OrderFlow"
    )
    reason_cache = {first: 1000.0}
    assert second in reason_cache


def test_unknown_side_reason_cooldown_key_preserves_legacy_behavior() -> None:
    assert (
        StrategyRunner._reason_order_cooldown_key(
            underlying="NIFTY", option_side="UNKNOWN", reason_key="OrderFlow"
        )
        == "NIFTY:OrderFlow"
    )


def test_live_scalping_cooldown_defaults_match_operator_comments(monkeypatch) -> None:
    for key in (
        "RUNNER_UNDERLYING_SIGNAL_COOLDOWN_SECONDS",
        "RUNNER_REASON_SIGNAL_COOLDOWN_SECONDS",
        "RUNNER_MAX_ORDER_ATTEMPTS_PER_MINUTE",
        "SIGNAL_REJECT_COOLDOWN_SECONDS",
    ):
        monkeypatch.delenv(key, raising=False)
    source = Path("src/nifty_scalper_bot/strategies/runner.py").read_text()
    assert 'RUNNER_UNDERLYING_SIGNAL_COOLDOWN_SECONDS", "20"' in source
    assert 'RUNNER_REASON_SIGNAL_COOLDOWN_SECONDS", "30"' in source
    assert 'RUNNER_MAX_ORDER_ATTEMPTS_PER_MINUTE", "5"' in source
    assert 'SIGNAL_REJECT_COOLDOWN_SECONDS", "15"' in source


def test_failed_entry_order_clears_submission_cooldowns_without_position() -> None:
    runner = object.__new__(StrategyRunner)
    runner._submitted_entry_order_context = {
        "OID1": {
            "symbol": "NFO:NIFTY2670724100PE",
            "underlying": "NIFTY",
            "underlying_reason_key": "NIFTY:PE:OrderFlow",
        }
    }
    runner._underlying_last_signal_ts = {"NIFTY": 1000.0}
    runner._reason_last_signal_ts = {"NIFTY:PE:OrderFlow": 1000.0}
    runner._position_manager = type("PM", (), {"has_open_position": lambda self, symbol: False})()
    runner._orchestrator = None
    runner._logger = type("Log", (), {"info": lambda *a, **k: None})()

    runner.notify_entry_order_failed(order_id="OID1", symbol="NFO:NIFTY2670724100PE", reason="rejected")

    assert runner._underlying_last_signal_ts == {}
    assert runner._reason_last_signal_ts == {}


def test_failed_entry_order_keeps_cooldowns_when_position_exists() -> None:
    runner = object.__new__(StrategyRunner)
    runner._submitted_entry_order_context = {
        "OID1": {
            "symbol": "NFO:NIFTY2670724100PE",
            "underlying": "NIFTY",
            "underlying_reason_key": "NIFTY:PE:OrderFlow",
        }
    }
    runner._underlying_last_signal_ts = {"NIFTY": 1000.0}
    runner._reason_last_signal_ts = {"NIFTY:PE:OrderFlow": 1000.0}
    runner._position_manager = type("PM", (), {"has_open_position": lambda self, symbol: True})()
    runner._orchestrator = None
    runner._logger = type("Log", (), {"info": lambda *a, **k: None})()

    runner.notify_entry_order_failed(order_id="OID1", symbol="NFO:NIFTY2670724100PE", reason="rejected")

    assert runner._underlying_last_signal_ts == {"NIFTY": 1000.0}
    assert runner._reason_last_signal_ts == {"NIFTY:PE:OrderFlow": 1000.0}


def test_order_failure_cooldown_rejection_uses_dedup_rollback_path() -> None:
    source = Path("src/nifty_scalper_bot/strategies/runner.py").read_text()
    assert 'reason="order_failure_cooldown_active")' in source
    assert 'return self._reject_signal_execution(symbol=base_symbol, trace_id=trace_id, reason="order_failure_cooldown_active")' not in source

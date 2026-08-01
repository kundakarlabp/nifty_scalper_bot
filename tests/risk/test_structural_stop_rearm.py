from __future__ import annotations

import time
from types import SimpleNamespace

from nifty_scalper_bot.execution.position_manager import PositionManager


def _signal(symbol: str, **metadata):
    return SimpleNamespace(symbol=symbol, metadata=metadata)


def _stopped_manager(monkeypatch, tmp_path) -> PositionManager:
    monkeypatch.setenv("STOP_LOSS_REENTRY_COOLDOWN_SECONDS", "1")
    pm = PositionManager(state_file=str(tmp_path / "positions.json"))
    pm.open_position(
        symbol="NFO:NIFTY2680424400PE",
        side="LONG",
        quantity=65,
        entry_price=93.35,
    )
    pm.close_position(
        "NFO:NIFTY2680424400PE",
        exit_price=91.40,
        reason="STOP_LOSS",
    )
    return pm


def test_time_alone_does_not_rearm_stopped_thesis(monkeypatch, tmp_path) -> None:
    pm = _stopped_manager(monkeypatch, tmp_path)
    pm._recent_stop_thesis["expires_epoch"] = time.time() - 1

    reason = pm.stop_reentry_block_reason(
        _signal("NFO:NIFTY2680424350PE")
    )

    assert reason == "stop-loss thesis awaiting newer setup candle"


def test_reused_setup_anchor_remains_blocked(monkeypatch, tmp_path) -> None:
    pm = _stopped_manager(monkeypatch, tmp_path)
    stopped_at = float(pm._recent_stop_thesis["stopped_at_epoch"])
    pm._recent_stop_thesis["expires_epoch"] = time.time() - 1

    reason = pm.stop_reentry_block_reason(
        _signal(
            "NFO:NIFTY2680424350PE",
            setup_candle_timestamp=stopped_at - 60,
        )
    )

    assert reason == "stop-loss thesis setup not rearmed"


def test_newer_setup_candle_rearms_after_minimum_cooldown(monkeypatch, tmp_path) -> None:
    pm = _stopped_manager(monkeypatch, tmp_path)
    stopped_at = float(pm._recent_stop_thesis["stopped_at_epoch"])
    pm._recent_stop_thesis["expires_epoch"] = time.time() - 1

    reason = pm.stop_reentry_block_reason(
        _signal(
            "NFO:NIFTY2680424350PE",
            setup_candle_timestamp=stopped_at + 60,
        )
    )

    assert reason is None
    assert pm._recent_stop_thesis is None


def test_new_setup_cannot_bypass_minimum_cooldown(monkeypatch, tmp_path) -> None:
    pm = _stopped_manager(monkeypatch, tmp_path)
    stopped_at = float(pm._recent_stop_thesis["stopped_at_epoch"])

    reason = pm.stop_reentry_block_reason(
        _signal(
            "NFO:NIFTY2680424350PE",
            setup_candle_timestamp=stopped_at + 60,
        )
    )

    assert "stop-loss thesis cooldown active" in reason


def test_opposite_option_side_is_not_blocked(monkeypatch, tmp_path) -> None:
    pm = _stopped_manager(monkeypatch, tmp_path)

    assert (
        pm.stop_reentry_block_reason(
            _signal("NFO:NIFTY2680424400CE")
        )
        is None
    )


def test_structural_lock_survives_restart_after_timer(monkeypatch, tmp_path) -> None:
    pm = _stopped_manager(monkeypatch, tmp_path)
    pm._recent_stop_thesis["expires_epoch"] = time.time() - 1
    pm.save_state()

    restarted = PositionManager(state_file=str(tmp_path / "positions.json"))

    assert restarted._recent_stop_thesis is not None
    assert restarted.stop_reentry_block_reason(
        _signal("NFO:NIFTY2680424350PE")
    ) == "stop-loss thesis awaiting newer setup candle"


def test_live_bracket_sl_reason_is_classified_as_stop() -> None:
    from nifty_scalper_bot.execution.position_risk_state_patch import _is_stop_reason

    assert _is_stop_reason("SL Hit (91.4 <= 92.00)") is True
    assert _is_stop_reason("HARD_SL_BREACH") is True
    assert _is_stop_reason("WATCHDOG_HARD_SL") is True
    assert _is_stop_reason("FORCED_SL_EXIT") is True
    assert _is_stop_reason("STOP_LOSS") is True
    assert _is_stop_reason("TP1 Hit (95.0)") is False
    assert _is_stop_reason("FINAL TP Hit (95.0)") is False
    assert _is_stop_reason("External/Manual Exit Detected") is False
    assert _is_stop_reason("EOD_FLATTEN") is False
    assert _is_stop_reason("SLIPPAGE_GUARD") is False
    assert _is_stop_reason("") is False


def test_record_stop_exit_latches_without_close_position(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("STOP_LOSS_REENTRY_COOLDOWN_SECONDS", "1")
    pm = PositionManager(state_file=str(tmp_path / "positions.json"))

    assert pm.record_stop_exit("NFO:NIFTY2680424400PE", "SL Hit (91.4 <= 92.00)")

    pm._recent_stop_thesis["expires_epoch"] = time.time() - 1
    assert (
        pm.stop_reentry_block_reason(_signal("NFO:NIFTY2680424350PE"))
        == "stop-loss thesis awaiting newer setup candle"
    )


def test_record_stop_exit_ignores_non_stop_exits(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("STOP_LOSS_REENTRY_COOLDOWN_SECONDS", "1")
    pm = PositionManager(state_file=str(tmp_path / "positions.json"))

    assert pm.record_stop_exit("NFO:NIFTY2680424400PE", "TP1 Hit (95.0)") is False
    assert pm.record_stop_exit("NFO:NIFTY2680424400PE", "EOD_FLATTEN") is False
    assert getattr(pm, "_recent_stop_thesis", None) is None


def test_bracket_exit_complete_latches_structural_stop(monkeypatch, tmp_path) -> None:
    import logging

    from nifty_scalper_bot.strategies.runner import StrategyRunner

    monkeypatch.setenv("STOP_LOSS_REENTRY_COOLDOWN_SECONDS", "1")
    pm = PositionManager(state_file=str(tmp_path / "positions.json"))
    symbol = "NFO:NIFTY2680424400PE"
    runner = SimpleNamespace(
        _logger=logging.getLogger("test.rearm"),
        _position_manager=pm,
        _strategy_manager=None,
        _bracket_manager=None,
        current_symbol=symbol,
        _normalize_symbol=lambda value: value,
        _notify_orchestrator_exit=lambda value: None,
        _clear_order_in_flight=lambda value: None,
        _release_entry_guards=lambda *a, **k: None,
    )

    StrategyRunner._on_bracket_exit_complete(
        runner,
        symbol,
        outcome={"symbol": symbol, "exit_reason": "SL Hit (91.4 <= 92.00)"},
    )

    assert pm._recent_stop_thesis is not None
    assert pm._recent_stop_thesis["option_side"] == "PE"
    assert pm._recent_stop_thesis["rearm_required"] is True

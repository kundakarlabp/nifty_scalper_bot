from __future__ import annotations

import logging

from nifty_scalper_bot.utils.alerts import AlertLogHandler


def _record(message: str, *, name: str = "nifty_scalper_bot.strategies.runner") -> logging.LogRecord:
    return logging.LogRecord(
        name=name,
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
        func="watchdog",
    )


def test_dynamic_strategy_stall_messages_share_one_semantic_cooldown() -> None:
    emitted: list[dict[str, str]] = []
    clock = iter([100.0, 220.0, 1001.0])
    handler = AlertLogHandler(
        emitted.append,
        repeat_window_seconds=300.0,
        persistent_repeat_window_seconds=900.0,
        clock=lambda: next(clock),
    )

    handler.emit(_record("Strategy evaluation stalled >120s (once per 120s)"))
    handler.emit(_record("Strategy evaluation stalled >241s (once per 120s)"))
    handler.emit(_record("Strategy evaluation stalled >900s (once per 120s)"))

    assert len(emitted) == 2
    assert emitted[0]["key"].endswith(":strategy_evaluation_stalled")
    assert emitted[1]["key"] == emitted[0]["key"]


def test_ticks_flowing_stall_and_1006_are_independently_throttled() -> None:
    emitted: list[dict[str, str]] = []
    times = iter([1.0, 2.0, 3.0, 4.0])
    handler = AlertLogHandler(
        emitted.append,
        persistent_repeat_window_seconds=900.0,
        clock=lambda: next(times),
    )

    handler.emit(_record("Strategy eval genuinely stalled while ticks flowing (>90s)"))
    handler.emit(_record("Strategy eval genuinely stalled while ticks flowing (>180s)"))
    handler.emit(
        _record(
            "WEBSOCKET_DEGRADED code=1006 reason=closing_handshake_timeout",
            name="nifty_scalper_bot.streaming.websocket_manager",
        )
    )
    handler.emit(
        _record(
            "WEBSOCKET_DEGRADED code=1006 reason=closing_handshake_timeout",
            name="nifty_scalper_bot.streaming.websocket_manager",
        )
    )

    assert len(emitted) == 2
    assert emitted[0]["key"].endswith(":strategy_eval_ticks_flowing_stalled")
    assert emitted[1]["key"].endswith(":websocket_degraded_1006")


def test_unmatched_warning_repeats_are_not_hidden() -> None:
    emitted: list[dict[str, str]] = []
    handler = AlertLogHandler(emitted.append, clock=lambda: 10.0)
    record = _record("Unexpected warning that needs every occurrence")

    handler.emit(record)
    handler.emit(record)

    assert len(emitted) == 2

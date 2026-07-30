"""event_loop_thread telemetry must be tri-state and never assert a guess.

Post-#943 audit, P2. TICK_STAGE_SLOW computed:

    event_loop_thread = thread_id == getattr(self, "_event_loop_thread_id", None)

`_event_loop_thread_id` was never assigned anywhere, so the getattr default of
None made the comparison unconditionally False: every record claimed
event_loop_thread=False, including records emitted from the loop thread.

Initialising the owner is not sufficient on its own. While the owner is
unestablished the equality test still yields False, which reports "unknown" as
"confirmed off-loop" -- the same confidently-wrong field. The value is
therefore tri-state:

    True    confirmed loop thread
    False   owner known, and this is a different thread
    None    owner not established (rendered "unknown" in the message)

These tests assert the emitted log record, not just internal state.
"""

from __future__ import annotations

import asyncio
import threading

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


def _mdm() -> MarketDataManager:
    return MarketDataManager(kite=None)


def _slow_records(caplog) -> list:
    """Emitted TICK_STAGE_SLOW log records (real emission path)."""
    return [
        r for r in caplog.records
        if getattr(r, "event", None) == "TICK_STAGE_SLOW"
    ]


_SEQ = [0]


def _emit_slow(mdm: MarketDataManager) -> None:
    # Unique symbol per emission: _log_slow_tick_stage throttles by key, so a
    # shared symbol would suppress later tests' records.
    _SEQ[0] += 1
    mdm._log_slow_tick_stage(
        stage="one_tick",
        symbol=f"NFO:NIFTY26JUN{24000 + _SEQ[0]}CE",
        duration_ms=999.0,
    )


# ---------- state 3: owner not established -> unknown, NOT False ----------


def test_unknown_owner_is_not_reported_as_confirmed_off_loop(caplog) -> None:
    """THE FIX: an unestablished owner must be None/unknown, never False."""
    import logging

    mdm = _mdm()
    assert mdm._event_loop_thread_id is None

    with caplog.at_level(logging.WARNING):
        _emit_slow(mdm)

    recs = _slow_records(caplog)
    assert recs, "expected a TICK_STAGE_SLOW record"
    assert recs[-1].event_loop_thread is None
    assert recs[-1].event_loop_thread is not False


# ---------- state 1: confirmed loop thread ----------


def test_confirmed_loop_thread_reports_true(caplog) -> None:
    import logging

    mdm = _mdm()

    async def _run() -> None:
        mdm.set_event_loop(asyncio.get_running_loop())
        _emit_slow(mdm)

    with caplog.at_level(logging.WARNING):
        asyncio.run(_run())

    assert _slow_records(caplog)[-1].event_loop_thread is True


# ---------- state 2: owner known, different thread ----------


def test_known_owner_on_other_thread_reports_false(caplog) -> None:
    """False must mean 'owner known and this is a different thread'."""
    import logging

    mdm = _mdm()
    mdm._event_loop_thread_id = threading.get_ident() + 1

    with caplog.at_level(logging.WARNING):
        _emit_slow(mdm)

    assert _slow_records(caplog)[-1].event_loop_thread is False


# ---------- owner establishment rules ----------


def test_thread_id_field_is_declared_not_an_accidental_default() -> None:
    mdm = _mdm()
    assert hasattr(mdm, "_event_loop_thread_id")
    assert mdm._event_loop_thread_id is None


def test_set_event_loop_off_thread_claims_no_ownership() -> None:
    """Wiring from another thread must leave the owner unestablished."""
    mdm = _mdm()
    loop = asyncio.new_event_loop()
    try:
        mdm.set_event_loop(loop)
        assert mdm._event_loop_thread_id is None
    finally:
        loop.close()


def test_message_renders_unknown_rather_than_none(caplog) -> None:
    """Operators must read 'unknown', not the literal None."""
    import logging

    mdm = _mdm()
    with caplog.at_level(logging.WARNING):
        _emit_slow(mdm)

    assert "event_loop_thread=unknown" in caplog.text
    assert "event_loop_thread=None" not in caplog.text

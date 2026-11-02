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
    # Give every emission a unique callback identity. Production has one
    # long-lived MDM instance, but tests create short-lived instances whose
    # Python object ids can be reused by the global log-throttle cache.
    _SEQ[0] += 1
    mdm._log_slow_tick_stage(
        stage="one_tick",
        symbol=f"NFO:NIFTY26JUN{24000 + _SEQ[0]}CE",
        duration_ms=999.0,
        callback=f"test_emit_{_SEQ[0]}",
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


def test_running_loop_attachment_records_owner_on_loop_thread() -> None:
    """Off-thread wiring must establish ownership from the scheduled callback."""
    mdm = _mdm()
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    callback_ran = threading.Event()
    try:
        mdm.set_event_loop(loop)
        loop.call_soon_threadsafe(callback_ran.set)
        assert callback_ran.wait(timeout=1.0)
        assert mdm._event_loop_thread_id == thread.ident
    finally:
        asyncio.run_coroutine_threadsafe(mdm.async_stop(), loop).result(timeout=1.0)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=1.0)
        loop.close()


def test_async_stop_detaches_event_loop_owner() -> None:
    """Shutdown must not retain ownership of a loop that is no longer attached."""

    async def _run() -> None:
        mdm = _mdm()
        mdm.set_event_loop(asyncio.get_running_loop())
        assert mdm._event_loop_thread_id == threading.get_ident()

        await mdm.async_stop()

        assert mdm._event_loop_thread_id is None
        assert mdm._main_loop is None

    asyncio.run(_run())


def test_message_renders_unknown_rather_than_none(caplog) -> None:
    """Operators must read 'unknown', not the literal None."""
    import logging

    mdm = _mdm()
    with caplog.at_level(logging.WARNING):
        _emit_slow(mdm)

    assert "event_loop_thread=unknown" in caplog.text
    assert "event_loop_thread=None" not in caplog.text


def test_periodic_tick_latency_summary_uses_real_all_tick_samples(caplog) -> None:
    import logging

    mdm = _mdm()
    for count in range(29):
        mdm._record_tick_latency("worker_await", float(count + 1))
    with caplog.at_level(logging.INFO, logger=mdm._logger.name):
        mdm._log_tick_latency_summary()
    assert not any(
        getattr(record, "event", None) == "TICK_LATENCY_SUMMARY"
        for record in caplog.records
    )

    mdm._record_tick_latency("worker_await", 30.0)
    mdm._record_tick_latency("one_tick", 11.0)
    mdm._record_tick_latency("candle_engine_on_tick", 3.0)
    mdm._record_tick_latency("tick_publication", 7.0)
    assert mdm._tick_latency_percentiles()["worker_await"]["samples"] == 30

    with caplog.at_level(logging.INFO, logger=mdm._logger.name):
        mdm._log_tick_latency_summary()
    records = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "TICK_LATENCY_SUMMARY"
    ]
    assert len(records) == 1
    record = records[0]
    assert record.sample_window == "last_512_per_stage"
    assert record.stage_percentiles["worker_await"]["samples"] == 30
    assert record.stage_percentiles["worker_await"]["p95_ms"] == 29.0
    assert record.stage_percentiles["worker_await"]["p99_ms"] == 30.0
    assert "worker_await:n=30" in record.getMessage()
    assert "one_tick:n=1" in record.getMessage()
    assert "tick_publication:n=1" in record.getMessage()


def test_latency_summary_exposes_cache_replay_and_bus_percentiles(caplog) -> None:
    """Summaries identify sub-stages without changing the trading gate."""
    import logging

    mdm = _mdm()
    for i in range(30):
        mdm._record_tick_latency("worker_await", float(i + 1))
        mdm._record_tick_latency("cache_store", 1.0)
        mdm._record_tick_latency("replay_capture", 2.0)
        mdm._record_tick_latency("bus_publish", 3.0)
        mdm._record_tick_latency("bar_projection", 4.0)
        mdm._record_tick_latency("closed_bar_pipeline_push", 5.0)
        mdm._record_tick_latency("closed_bar_subscriber_callback", 6.0)
    with caplog.at_level(logging.INFO, logger=mdm._logger.name):
        mdm._log_tick_latency_summary()
    summaries = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "TICK_LATENCY_SUMMARY"
    ]
    assert len(summaries) == 1
    assert summaries[0].stage_percentiles["cache_store"]["p99_ms"] == 1.0
    assert summaries[0].stage_percentiles["replay_capture"]["p99_ms"] == 2.0
    assert summaries[0].stage_percentiles["bus_publish"]["p99_ms"] == 3.0
    assert summaries[0].stage_percentiles["bar_projection"]["p99_ms"] == 4.0
    assert summaries[0].stage_percentiles["closed_bar_pipeline_push"]["p99_ms"] == 5.0
    assert (
        summaries[0].stage_percentiles["closed_bar_subscriber_callback"]["p99_ms"]
        == 6.0
    )
    assert "bus_publish:n=30" in summaries[0].getMessage()


def test_latency_warmup_is_observable_but_not_a_percentile_summary(caplog) -> None:
    """A partially sampled worker emits provenance, not fake percentiles."""
    import logging

    mdm = _mdm()
    mdm._record_tick_latency("worker_await", 10.0)
    with caplog.at_level(logging.INFO, logger=mdm._logger.name):
        mdm._log_tick_latency_summary()
    warmup = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "TICK_LATENCY_WARMUP"
    ]
    assert len(warmup) == 1
    assert warmup[0].worker_samples == 1
    assert not any(
        getattr(record, "event", None) == "TICK_LATENCY_SUMMARY"
        for record in caplog.records
    )


def test_closed_bar_callback_samples_real_publish_path(monkeypatch) -> None:
    """Measure canonical bar fanout without broker calls or second bar owner."""
    mdm = _mdm()
    called = []
    monkeypatch.setattr(mdm, "_push_bar_to_pipeline", lambda bar: None)
    mdm._bar_subscribers = [lambda bar: called.append(bar["symbol"])]

    mdm._publish_closed_bar(
        {"symbol": "NSE:NIFTY", "source": "ws_candle", "close": 24000.0}
    )

    assert called == ["NSE:NIFTY"]
    stages = mdm._tick_latency_percentiles()
    assert stages["closed_bar_pipeline_push"]["samples"] == 1
    assert stages["closed_bar_subscriber_callback"]["samples"] == 1

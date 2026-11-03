from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from nifty_scalper_bot.infra.daily_log_archive import DailyLogArchiver

IST = ZoneInfo("Asia/Kolkata")


def test_archive_snapshots_complete_session_and_finalizes_after_close() -> None:
    calls = []

    def read_logs(start, end):
        assert start.hour == 9 and start.minute == 15
        assert end.hour == 15 and end.minute == 30
        return "line one\nline two"

    def transport(url, payload, timeout):
        calls.append((url, payload, timeout))
        return {"ok": True}

    archiver = DailyLogArchiver(
        endpoint_url="https://example.test/archive",
        transport=transport,
        log_reader=read_logs,
    )
    result = archiver.archive_once(datetime(2026, 9, 24, 16, 0, tzinfo=IST))

    assert result == {"archived": True, "line_count": 2, "finalized": True}
    payload = calls[0][1]
    assert payload["trading_date"] == "2026-09-24"
    assert payload["finalized"] is True
    assert payload["log_text"] == "line one\nline two"


def test_archive_uses_canonical_runtime_build_sha(monkeypatch) -> None:
    calls = []
    monkeypatch.setenv("GIT_COMMIT_SHA", "abc123")

    archiver = DailyLogArchiver(
        endpoint_url="https://example.test/archive",
        transport=lambda _url, payload, _timeout: calls.append(dict(payload))
        or {"ok": True},
        log_reader=lambda _start, _end: "line one",
    )

    archiver.archive_once(datetime(2026, 9, 24, 16, 0, tzinfo=IST))

    assert calls[0]["build_sha"] == "abc123"


def test_archive_finalized_session_only_once_per_process() -> None:
    calls = []
    reads = []

    archiver = DailyLogArchiver(
        endpoint_url="https://example.test/archive",
        transport=lambda _url, payload, _timeout: calls.append(dict(payload))
        or {"ok": True},
        log_reader=lambda start, end: reads.append((start, end)) or "line one",
    )

    first = archiver.archive_once(datetime(2026, 9, 24, 16, 0, tzinfo=IST))
    second = archiver.archive_once(datetime(2026, 9, 24, 16, 5, tzinfo=IST))

    assert first["archived"] is True
    assert second == {"archived": False, "reason": "already_finalized"}
    assert len(reads) == 1
    assert len(calls) == 1


def test_archive_skips_before_market_open() -> None:
    def fail_read(*_args):
        raise AssertionError("must not read")

    archiver = DailyLogArchiver(
        endpoint_url="https://example.test/archive",
        transport=lambda *_args: {"ok": True},
        log_reader=fail_read,
    )

    result = archiver.archive_once(datetime(2026, 9, 24, 9, 0, tzinfo=IST))

    assert result == {"archived": False, "reason": "outside_session"}


def test_archive_includes_bounded_startup_audit_without_protection_claim() -> None:
    calls = []
    journal = (
        "[2026-10-23 09:49:33 IST] STARTUP_PHASE phase=create_context\n"
        "[2026-10-23 09:49:35 IST] STARTUP_WS_START_DONE phase=spot_first\n"
        "[2026-10-23 09:49:36 IST] BROKER_ORDER_RECONCILE_OK seen=2\n"
        "[2026-10-23 09:50:04 IST] RECONCILE_COMPLETE\n"
        "[2026-10-23 09:50:04 IST] STARTUP_MODE_TRANSITION "
        "from=DATA_WARMUP to=LIVE_READY\n"
        "access_token=DO_NOT_REPLICATE_IN_SUMMARY"
    )
    archiver = DailyLogArchiver(
        endpoint_url="https://example.test/archive",
        transport=lambda _url, payload, _timeout: calls.append(payload) or {"ok": True},
        log_reader=lambda _start, _end: journal,
    )
    assert (
        archiver.archive_once(datetime(2026, 10, 23, 16, 0, tzinfo=IST))["archived"]
        is True
    )
    audit = calls[0]["startup_audit"]
    assert audit["observed_startup_count"] == 1
    assert audit["startups_after_0930_count"] == 1
    assert audit["max_startup_to_ready_seconds"] == 31
    assert audit["protective_stop_verification"] == "not_established_from_log_markers"
    assert "DO_NOT_REPLICATE" not in str(audit)


def test_research_waits_for_settle_and_replicates_summary() -> None:
    calls = []
    research_calls = []

    def research(day, journal):
        research_calls.append((day, journal))
        return {
            "date_ist": day.isoformat(),
            "status": "analysed",
            "blocker": None,
            "replay_sha256": "a" * 64,
            "tradable_ws_quote_ticks": 4,
            "independent_setup_count": 0,
            "sampled_observations": [],
            "sampled_strata": [],
            "structural_strata": [],
            "volume_parity": {
                "observations": 2,
                "positive_volume_matches": 2,
            },
        }

    archiver = DailyLogArchiver(
        endpoint_url="https://example.test/archive",
        log_reader=lambda _start, _end: "normal journal entry",
        transport=lambda _url, payload, _timeout: calls.append(dict(payload))
        or {"ok": True},
        research_runner=research,
    )
    before = archiver.archive_once(datetime(2026, 11, 3, 15, 31, tzinfo=IST))
    after = archiver.archive_once(datetime(2026, 11, 3, 15, 35, tzinfo=IST))
    assert before["finalized"] is False
    assert after["finalized"] is True
    assert len(research_calls) == 1
    assert "RESEARCH_SESSION_EVIDENCE " not in calls[0]["log_text"]
    assert "RESEARCH_SESSION_EVIDENCE " in calls[1]["log_text"]
    assert calls[1]["build_sha"] is not None
    assert calls[1]["line_count"] == 2
    assert archiver.archive_once(datetime(2026, 11, 3, 15, 40, tzinfo=IST)) == {
        "archived": False,
        "reason": "already_finalized",
    }

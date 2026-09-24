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


def test_archive_skips_before_market_open() -> None:
    archiver = DailyLogArchiver(
        endpoint_url="https://example.test/archive",
        transport=lambda *_args: {"ok": True},
        log_reader=lambda *_args: (_ for _ in ()).throw(\n            AssertionError("must not read")\n        ),
    )

    result = archiver.archive_once(datetime(2026, 9, 24, 9, 0, tzinfo=IST))

    assert result == {"archived": False, "reason": "outside_session"}

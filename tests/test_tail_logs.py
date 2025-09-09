from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.main import _tail_logs, _make_cmd_handler


def test_tail_logs_reads_last_lines(tmp_path: Path) -> None:
    log_file = tmp_path / "trading_bot.log"
    log_file.write_text("\n".join(["l1", "l2", "l3"]))
    assert _tail_logs(2, path=str(log_file)) == ["l2", "l3"]


def test_tail_logs_handles_error(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        lines = _tail_logs(5, path=str(tmp_path / "missing.log"))
    assert lines == []
    assert any("Failed to tail logs" in r.message for r in caplog.records)


def test_make_cmd_handler_executes_actions() -> None:
    calls: list[str] = []

    class Runner:
        def pause(self) -> None:
            calls.append("pause")

        def resume(self) -> None:
            calls.append("resume")

    handler = _make_cmd_handler(Runner())
    handler("/pause", "")
    handler("/resume", "")
    handler("/unknown", "")
    assert calls == ["pause", "resume"]

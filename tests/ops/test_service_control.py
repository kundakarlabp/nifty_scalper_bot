from __future__ import annotations

from types import SimpleNamespace

from nifty_scalper_bot.ops import service_control as sc


def test_restart_service_rejects_invalid_name() -> None:
    result = sc.restart_service("bad service; rm -rf /", action="restart_bot")
    assert result.ok is False
    assert result.action == "restart_bot"
    assert result.command == ()


def test_restart_service_uses_bounded_systemctl_command(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_popen(command, **kwargs):
        calls.append(list(command))
        assert kwargs["start_new_session"] is True
        return SimpleNamespace(pid=123)

    monkeypatch.setattr(sc.subprocess, "Popen", fake_popen)

    result = sc.restart_service("niftybot", action="restart_bot")

    assert result.ok is True
    assert calls == [["sudo", "systemctl", "restart", "--no-block", "niftybot"]]


def test_memory_snapshot_parses_proc_meminfo(tmp_path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:        1000000 kB\n"
        "MemAvailable:     250000 kB\n"
        "SwapTotal:        500000 kB\n"
        "SwapFree:         400000 kB\n",
        encoding="utf-8",
    )

    snapshot = sc.memory_snapshot(meminfo)

    assert snapshot["available"] is True
    assert snapshot["mem_used_pct"] == 75.0
    assert snapshot["swap_used_mb"] > 0

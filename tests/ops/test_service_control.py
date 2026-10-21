from __future__ import annotations

from types import SimpleNamespace

from nifty_scalper_bot.ops import service_control as sc


def test_restart_service_rejects_invalid_name() -> None:
    result = sc.restart_service("bad service; rm -rf /", action="restart_bot")
    assert result.ok is False
    assert result.action == "restart_bot"
    assert result.command == ()


def test_restart_service_requires_systemd_acceptance(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        assert kwargs["timeout"] == 8
        assert kwargs["capture_output"] is True
        assert kwargs["check"] is False
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(sc.subprocess, "run", fake_run)
    result = sc.restart_service(
        "niftybot-autodeploy.service", action="restart_deployer"
    )

    assert result.ok is True
    assert calls == [
        [
            "sudo",
            "-n",
            "systemctl",
            "restart",
            "--no-block",
            "niftybot-autodeploy.service",
        ]
    ]


def test_restart_service_rejects_failed_sudo(monkeypatch) -> None:
    monkeypatch.setattr(
        sc.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr="private error"),
    )
    result = sc.restart_service("niftybot", action="restart_bot")
    assert result.ok is False
    assert result.message == "systemd rejected restart (exit 1)"
    assert "private error" not in result.message


def test_restart_service_timeout_fails_closed(monkeypatch) -> None:
    def timeout(*args, **kwargs):
        raise TimeoutError("service restart timed out")

    monkeypatch.setattr(sc.subprocess, "run", timeout)
    result = sc.restart_service("niftybot", action="restart_bot")
    assert result.ok is False
    assert "TimeoutError" in result.message


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


def test_queued_restart_creates_marker_and_requests_only_deployer(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("NIFTYBOT_CONFIG_DIR", str(tmp_path))
    calls = []
    monkeypatch.setattr(
        sc,
        "restart_deployer",
        lambda: calls.append("autodeploy")
        or sc.ControlResult(
            True, "restart_deployer", "systemd accepted", ("systemctl",)
        ),
    )
    result = sc.queue_validated_restart()

    assert result.ok is True
    assert result.action == "restart_bot"
    assert "queued" in result.message.lower()
    assert calls == ["autodeploy"]
    assert (tmp_path / "auto_release_restart_pending").is_file()


def test_queued_restart_rejects_unwritable_state(monkeypatch, tmp_path) -> None:
    missing = tmp_path / "does-not-exist"
    monkeypatch.setenv("NIFTYBOT_CONFIG_DIR", str(missing))
    monkeypatch.setattr(
        sc,
        "restart_deployer",
        lambda: (_ for _ in ()).throw(
            AssertionError("must never invoke deployer without marker")
        ),
    )
    result = sc.queue_validated_restart()

    assert not result.ok
    assert "marker" in result.message.lower()


def test_queued_restart_reports_dispatch_failure_without_claiming_restart(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("NIFTYBOT_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(
        sc,
        "restart_deployer",
        lambda: sc.ControlResult(False, "restart_deployer", "systemd rejected"),
    )

    result = sc.queue_validated_restart()

    assert result.ok is False
    assert "systemd rejected" in result.message
    assert (tmp_path / "auto_release_restart_pending").exists()

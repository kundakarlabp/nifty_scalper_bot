"""Research requests must be bounded, single-flight and honest about evidence."""

import json
from datetime import date

import pytest

from nifty_scalper_bot.ops.research_jobs import validate_request


def test_research_request_uses_completed_dates_and_no_user_command():
    request = validate_request({"id": "request-1", "days": 30}, today=date(2026, 10, 2))
    assert request["end"] == "2026-10-01"
    assert request["start"] == "2026-09-02"
    for payload in (
        {"id": "../escape", "days": 30},
        {"id": "safe", "days": 366},
        {"id": "safe", "command": "touch /tmp/order"},
        {"id": "safe", "days": True},
    ):
        with pytest.raises(ValueError):
            validate_request(payload, today=date(2026, 10, 2))


def test_launch_is_idempotent_and_forces_offline_order_flags(tmp_path, monkeypatch):
    from nifty_scalper_bot.ops.research_jobs import start_job

    launches = []
    monkeypatch.setattr(
        "nifty_scalper_bot.ops.research_jobs.subprocess.Popen",
        lambda command, **kwargs: launches.append((command, kwargs)),
    )
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    payload = {"id": "one-job", "days": 30}
    first = start_job(tmp_path, payload)
    second = start_job(tmp_path, payload)
    assert first == second
    assert len(launches) == 1
    command, options = launches[0]
    assert command[1] == str(tmp_path / "scripts/run_research_job.py")
    assert options["env"]["ENABLE_LIVE"] == "false"
    assert options["env"]["ORDERS__ENABLE_LIVE"] == "false"
    assert options["env"]["EXECUTION_MODE"] == "SHADOW"
    assert options["pass_fds"]
    assert first["backtest_completed"] is False
    assert __import__("os").environ["EXECUTION_MODE"] == "LIVE"


def test_overlapping_job_is_rejected(tmp_path):
    import fcntl

    from nifty_scalper_bot.ops.research_jobs import start_job

    directory = tmp_path / "data/research"
    directory.mkdir(parents=True)
    with (directory / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert start_job(tmp_path, {"id": "second"})["state"] == "busy"
    assert not (directory / "second").exists()


def test_failed_launch_is_explicit_without_secret_exception_text(tmp_path, monkeypatch):
    from nifty_scalper_bot.ops.research_jobs import read_status, start_job

    def fail(*args, **kwargs):
        raise OSError("SECRET_TOKEN")

    monkeypatch.setattr("nifty_scalper_bot.ops.research_jobs.subprocess.Popen", fail)
    assert start_job(tmp_path, {"id": "failed-job"})["state"] == "failed"
    status = read_status(tmp_path)
    assert status["error_type"] == "OSError"
    assert "SECRET_TOKEN" not in json.dumps(status)
    assert status["backtest_completed"] is False


def test_dashboard_rejects_cross_site_start(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from nifty_scalper_bot import admin_dashboard

    monkeypatch.setattr(admin_dashboard, "APP_DIR", tmp_path)
    app = FastAPI()
    app.include_router(admin_dashboard.router)
    client = TestClient(app)
    response = client.post(
        "/admin/research/start", headers={"origin": "https://attacker.invalid"}
    )
    assert response.status_code == 403
    assert not (tmp_path / "data/research").exists()
    assert client.get("/admin/research/status").json()["state"] == "not_requested"


def test_worker_reports_real_collection_as_blocked_not_backtest_success(
    tmp_path, monkeypatch
):
    import io
    from types import SimpleNamespace

    from scripts import run_research_job as worker

    class ReadOnlyBroker:
        def instruments(self, exchange):
            return [
                {
                    "exchange": "NFO",
                    "name": "NIFTY",
                    "tradingsymbol": "NIFTY26OCTFUT",
                    "instrument_type": "FUT",
                    "expiry": "2026-10-27",
                }
            ]

        def close(self):
            pass

    captures = []
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    monkeypatch.setattr(
        worker,
        "urlopen",
        lambda *args, **kwargs: io.StringIO(
            '{"selected":{"ce":"NFO:CE","pe":"NFO:PE"}}'
        ),
    )
    monkeypatch.setattr(
        "nifty_scalper_bot.data.rest.zerodha_client.ZerodhaKiteClient",
        ReadOnlyBroker,
    )
    monkeypatch.setattr(
        "nifty_scalper_bot.instruments.active_contracts.resolve_active_nifty_future_from_instruments",
        lambda rows: SimpleNamespace(symbol="NFO:NIFTY26OCTFUT"),
    )
    monkeypatch.setattr(
        "nifty_scalper_bot.config.paths.get_data_dir", lambda: tmp_path / "no-ledger"
    )

    def collect(client, symbols, start, end, outdir):
        captures.append((symbols, start, end))
        return {"saved_requests": 4, "empty_requests": [], "failed_requests": []}

    monkeypatch.setattr("scripts.archive_kite_history.archive_history", collect)
    request = validate_request({"id": "real-job"}, today=date(2026, 10, 2))
    result = worker.run_worker(request, tmp_path / "empty.env")
    assert captures[0][0] == ["NSE:NIFTY 50", "NFO:NIFTY26OCTFUT", "NFO:CE", "NFO:PE"]
    assert result["state"] == "blocked"
    assert result["backtest_completed"] is False
    assert result["blocker"] == "current_bot_offline_replay_adapter_unavailable"


def test_updater_waits_for_worker_before_oneshot_service_exits(tmp_path, monkeypatch):
    from nifty_scalper_bot.ops.research_jobs import start_job, write_json

    waited = []

    class Worker:
        def wait(self, timeout):
            waited.append(timeout)
            status = {"id": "oneshot", "state": "blocked", "backtest_completed": False}
            write_json(tmp_path / "data/research/oneshot/status.json", status)
            write_json(tmp_path / "data/research/latest.json", status)
            return 0

    monkeypatch.setattr(
        "nifty_scalper_bot.ops.research_jobs.subprocess.Popen",
        lambda *args, **kwargs: Worker(),
    )
    result = start_job(tmp_path, {"id": "oneshot"}, wait=True)
    assert waited == [1800]
    assert result["state"] == "blocked"


@pytest.mark.parametrize("timeout", [True, False])
def test_updater_records_worker_timeout_or_unexpected_exit(
    tmp_path, monkeypatch, timeout
):
    import subprocess

    from nifty_scalper_bot.ops.research_jobs import start_job

    killed = []

    class Worker:
        def wait(self, timeout=None):
            if timeout is not None and globals_timeout:
                raise subprocess.TimeoutExpired("fixed-worker", timeout)
            return 1

        def kill(self):
            killed.append(True)

    globals_timeout = timeout
    monkeypatch.setattr(
        "nifty_scalper_bot.ops.research_jobs.subprocess.Popen",
        lambda *args, **kwargs: Worker(),
    )
    result = start_job(tmp_path, {"id": "dead-worker"}, wait=True)
    assert result["state"] == "failed"
    assert result["error_type"] == (
        "WorkerTimeout" if timeout else "WorkerExitedWithoutResult"
    )
    assert bool(killed) is timeout

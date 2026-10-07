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


def test_launch_is_idempotent_after_terminal_result_and_forces_offline_flags(
    tmp_path, monkeypatch
):
    from nifty_scalper_bot.ops.research_jobs import start_job, write_json

    launches = []
    monkeypatch.setattr(
        "nifty_scalper_bot.ops.research_jobs.subprocess.Popen",
        lambda command, **kwargs: launches.append((command, kwargs)),
    )
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    payload = {"id": "one-job", "days": 30}
    first = start_job(tmp_path, payload)
    terminal = {
        **first,
        "state": "completed",
        "backtest_completed": True,
        "requested_work_completed": True,
    }
    write_json(tmp_path / "data/research/one-job/status.json", terminal)
    write_json(tmp_path / "data/research/latest.json", terminal)

    second = start_job(tmp_path, payload)

    assert second == terminal
    assert len(launches) == 1
    command, options = launches[0]
    assert command[1] == str(tmp_path / "scripts/run_research_job.py")
    assert options["env"]["ENABLE_LIVE"] == "false"
    assert options["env"]["ORDERS__ENABLE_LIVE"] == "false"
    assert options["env"]["EXECUTION_MODE"] == "SHADOW"
    assert options["pass_fds"]
    assert first["launch_attempt"] == 1
    assert first["backtest_completed"] is False
    assert __import__("os").environ["EXECUTION_MODE"] == "LIVE"


def test_systemd_launcher_uses_independent_transient_service(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from nifty_scalper_bot.ops.research_jobs import start_job

    calls = []
    monkeypatch.setenv("BOT_RESEARCH_LAUNCHER", "systemd")
    monkeypatch.setattr(
        "nifty_scalper_bot.ops.research_jobs._current_revision",
        lambda _root: "abc123def456",
    )

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command[0] == "systemctl":
            return SimpleNamespace(
                returncode=0,
                stdout="LoadState=not-found\nActiveState=inactive\n",
                stderr="",
            )
        if command[:3] == ["sudo", "-n", "systemd-run"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(command)

    monkeypatch.setattr("nifty_scalper_bot.ops.research_jobs.subprocess.run", run)
    monkeypatch.setattr(
        "nifty_scalper_bot.ops.research_jobs.subprocess.Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("systemd mode must not spawn an admin child")
        ),
    )

    result = start_job(
        tmp_path,
        {"id": "systemd-job", "days": 30, "mode": "components"},
    )

    assert result["state"] == "queued"
    assert result["launcher"] == "systemd"
    assert result["worker_unit"] == "niftybot-research-worker.service"
    assert result["launch_revision"] == "abc123def456"
    systemd_prefix = ["sudo", "-n", "systemd-run"]
    launch = next(command for command, _ in calls if command[:3] == systemd_prefix)
    assert "--unit=niftybot-research-worker.service" in launch
    assert "--collect" in launch
    assert "--service-type=exec" in launch
    assert "--property=RuntimeMaxSec=2100s" in launch
    assert "--property=CPUWeight=5" in launch
    assert "--property=IOWeight=10" in launch
    assert "--property=OOMScoreAdjust=500" in launch
    assert "--setenv=ENABLE_LIVE=false" in launch
    assert "--setenv=ORDERS__ENABLE_LIVE=false" in launch
    assert "--setenv=EXECUTION_MODE=SHADOW" in launch
    assert str(tmp_path / "scripts/run_research_job.py") in launch


def test_systemd_active_worker_keeps_same_nonterminal_job_live(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from nifty_scalper_bot.ops.research_jobs import start_job, write_json

    monkeypatch.setenv("BOT_RESEARCH_LAUNCHER", "systemd")
    payload = {"id": "live-job", "days": 30, "mode": "components"}
    existing = {
        "id": "live-job",
        "days": 30,
        "mode": "components",
        "state": "collecting",
        "stage": "history",
        "backtest_completed": False,
        "component_backtest_completed": False,
        "runtime_replay_completed": False,
        "requested_work_completed": False,
        "launch_attempt": 1,
        "launcher": "systemd",
        "worker_unit": "niftybot-research-worker.service",
    }
    write_json(tmp_path / "data/research/live-job/status.json", existing)
    write_json(tmp_path / "data/research/latest.json", existing)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert command[0] == "systemctl"
        return SimpleNamespace(
            returncode=0,
            stdout="LoadState=loaded\nActiveState=active\n",
            stderr="",
        )

    monkeypatch.setattr("nifty_scalper_bot.ops.research_jobs.subprocess.run", run)

    result = start_job(tmp_path, payload)

    assert result == existing
    assert all(command[:3] != ["sudo", "-n", "systemd-run"] for command in calls)


def test_systemd_active_worker_rejects_overlapping_request(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from nifty_scalper_bot.ops.research_jobs import start_job

    monkeypatch.setenv("BOT_RESEARCH_LAUNCHER", "systemd")
    monkeypatch.setattr(
        "nifty_scalper_bot.ops.research_jobs.subprocess.run",
        lambda command, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="LoadState=loaded\nActiveState=activating\n",
            stderr="",
        ),
    )

    result = start_job(tmp_path, {"id": "second-systemd-job", "mode": "components"})

    assert result["state"] == "busy"
    assert result["worker_unit"] == "niftybot-research-worker.service"
    assert not (tmp_path / "data/research/second-systemd-job").exists()


def test_systemd_launcher_state_failure_fails_closed(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from nifty_scalper_bot.ops.research_jobs import read_status, start_job

    monkeypatch.setenv("BOT_RESEARCH_LAUNCHER", "systemd")
    monkeypatch.setattr(
        "nifty_scalper_bot.ops.research_jobs.subprocess.run",
        lambda command, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="private host detail",
        ),
    )

    result = start_job(tmp_path, {"id": "systemd-unavailable", "mode": "components"})

    assert result["state"] == "failed"
    assert result["error_code"] == "research_launcher_state_unavailable"
    assert "private host detail" not in json.dumps(result)
    assert read_status(tmp_path) == result


@pytest.mark.parametrize("stale_state", ["queued", "collecting"])
def test_stale_nonterminal_job_fails_closed_without_relaunch(
    tmp_path, monkeypatch, stale_state
):
    from nifty_scalper_bot.ops.research_jobs import start_job, write_json

    payload = {"id": "stale-job", "days": 30, "mode": "components"}
    stale = {
        "id": "stale-job",
        "days": 30,
        "mode": "components",
        "start": "2026-09-07",
        "end": "2026-10-06",
        "state": stale_state,
        "stage": "history",
        "backtest_completed": False,
        "component_backtest_completed": False,
        "runtime_replay_completed": False,
        "requested_work_completed": False,
        "launch_attempt": 1,
    }
    status_file = tmp_path / "data/research/stale-job/status.json"
    write_json(status_file, stale)
    write_json(tmp_path / "data/research/latest.json", stale)
    launches = []
    monkeypatch.setattr(
        "nifty_scalper_bot.ops.research_jobs.subprocess.Popen",
        lambda command, **kwargs: launches.append((command, kwargs)),
    )

    result = start_job(tmp_path, payload)

    assert launches == []
    assert result["state"] == "failed"
    assert result["error_type"] == "WorkerExitedWithoutResult"
    assert result["error_code"] == "stale_research_worker"
    assert result["stale_state"] == stale_state
    assert result["launch_attempt"] == 1
    persisted = json.loads(status_file.read_text())
    assert persisted == result
    assert json.loads((tmp_path / "data/research/latest.json").read_text()) == result


def test_reused_request_id_with_different_definition_is_rejected(tmp_path):
    from nifty_scalper_bot.ops.research_jobs import start_job, write_json

    existing = {
        "id": "immutable-job",
        "days": 30,
        "mode": "components",
        "state": "completed",
        "backtest_completed": True,
    }
    write_json(tmp_path / "data/research/immutable-job/status.json", existing)

    with pytest.raises(ValueError, match="research_request_id_conflict"):
        start_job(
            tmp_path,
            {"id": "immutable-job", "days": 14, "mode": "components"},
        )


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


def test_admin_manifest_poll_launches_exact_fixed_request(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from nifty_scalper_bot import admin_dashboard
    from nifty_scalper_bot.ops import research_jobs

    manifest = tmp_path / "deploy/research_request.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({"id": "manifest-job", "days": 14, "mode": "components"}),
        encoding="utf-8",
    )
    launched = []

    def launch(root, payload, **kwargs):
        launched.append((root, payload, kwargs))
        return {
            "id": payload["id"],
            "state": "queued",
            "backtest_completed": False,
        }

    monkeypatch.setattr(admin_dashboard, "APP_DIR", tmp_path)
    monkeypatch.setattr(research_jobs, "start_job", launch)
    app = FastAPI()
    app.include_router(admin_dashboard.router)
    client = TestClient(app)

    response = client.post(
        "/admin/research/poll-manifest",
        headers={"origin": "http://testserver"},
    )

    assert response.status_code == 202
    assert launched == [
        (
            tmp_path,
            {"id": "manifest-job", "days": 14, "mode": "components"},
            {},
        )
    ]


def test_admin_research_report_exposes_vwap_comparison(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from nifty_scalper_bot import admin_dashboard
    from nifty_scalper_bot.ops.research_jobs import write_json

    monkeypatch.setattr(admin_dashboard, "APP_DIR", tmp_path)
    status = {
        "id": "report-job",
        "days": 30,
        "mode": "components",
        "state": "completed",
        "backtest_completed": True,
    }
    write_json(tmp_path / "data/research/latest.json", status)
    write_json(
        tmp_path / "data/research/report-job/vwap_comparison.json",
        {
            "scope": "bounded_vwap_active_contract_component_comparison",
            "selection": {"promotion_eligible": False},
        },
    )
    app = FastAPI()
    app.include_router(admin_dashboard.router)
    client = TestClient(app)

    response = client.get("/admin/research/report")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"]["id"] == "report-job"
    assert payload["vwap_comparison"]["selection"]["promotion_eligible"] is False


@pytest.mark.parametrize("ledger_error", [False, True])
def test_worker_completes_component_research_without_claiming_live_parity(
    tmp_path, monkeypatch, ledger_error
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

    def collect(client, symbols, start, end, outdir, **kwargs):
        captures.append((symbols, start, end))
        return {"saved_requests": 4, "empty_requests": [], "failed_requests": []}

    monkeypatch.setattr("scripts.archive_kite_history.archive_history", collect)
    monkeypatch.setattr(
        "nifty_scalper_bot.backtesting.strategy_research.run_archived_research",
        lambda directory: {
            "scope": "active_contract_strategy_components",
            "evidence_label": "RESEARCH_CANDIDATE",
            "coverage": {},
            "scenarios": [],
        },
    )
    monkeypatch.setattr(
        "nifty_scalper_bot.backtesting.strategy_research.run_orb_comparison",
        lambda directory: {
            "candidates": [],
            "selection": {"promotion_eligible": False},
        },
    )
    monkeypatch.setattr(
        "nifty_scalper_bot.backtesting.strategy_research.run_vwap_comparison",
        lambda directory: {
            "candidates": [],
            "selection": {"promotion_eligible": False},
        },
    )

    def blocked_runtime(root, request):
        in_progress = json.loads(
            (tmp_path / "data/research/real-job/status.json").read_text()
        )
        assert in_progress["state"] == "collecting"
        assert in_progress["stage"] == "runtime_replay"
        assert in_progress["component_backtest_completed"] is True
        assert in_progress["requested_work_completed"] is False
        return {
            "scope": "production_composition_recorded_feed_replay",
            "state": "blocked",
            "completed_sessions": 0,
            "failed_sessions": 0,
            "available_sessions_fully_processed": False,
        }

    monkeypatch.setattr(worker, "run_recorded_replays", blocked_runtime)
    if ledger_error:
        journal = tmp_path / "no-ledger/trades.db"
        journal.parent.mkdir()
        journal.touch()
        monkeypatch.setattr(
            "scripts.reporting.analyze_completed_trades.load_trade_ledger_rows",
            lambda path: [],
        )

        def reject_costs(*args, **kwargs):
            raise ValueError(
                "broker-calculated costs required for canonical research "
                "dataset: private-id"
            )

        monkeypatch.setattr(
            "scripts.reporting.analyze_completed_trades.build_analysis", reject_costs
        )
    request = validate_request({"id": "real-job"}, today=date(2026, 10, 2))
    result = worker.run_worker(request, tmp_path / "empty.env")
    assert captures[0][0] == ["NSE:NIFTY 50", "NFO:NIFTY26OCTFUT", "NFO:CE", "NFO:PE"]
    assert result["state"] == "partial"
    assert result["backtest_completed"] is True
    assert result["component_backtest_completed"] is True
    assert result["requested_work_completed"] is False
    assert result["runtime_replay_completed"] is False
    assert result["runtime_replay"]["state"] == "blocked"
    assert result["orb_comparison"]["selection"]["promotion_eligible"] is False
    assert (tmp_path / "data/research/real-job/orb_comparison.json").is_file()
    assert result["vwap_comparison"]["selection"]["promotion_eligible"] is False
    assert (tmp_path / "data/research/real-job/vwap_comparison.json").is_file()
    assert result["live_equivalent"] is False
    assert result["backtest_scope"] == "active_contract_strategy_components"
    assert (tmp_path / "data/research/real-job/strategy_bar_research.json").is_file()
    if ledger_error:
        assert result["ledger_analysis_blocker"] == "ledger_requires_verified_costs"
        assert result["coverage"]["saved_requests"] == 4
        assert "private-id" not in json.dumps(result)


@pytest.mark.parametrize(
    ("mode", "expected_timeout"),
    [("all", 3600), ("components", 2100), ("runtime", 1800)],
)
def test_updater_waits_with_mode_aware_budget(
    tmp_path, monkeypatch, mode, expected_timeout
):
    from nifty_scalper_bot.ops.research_jobs import start_job, write_json

    waited = []

    class Worker:
        def wait(self, timeout):
            waited.append(timeout)
            status = {
                "id": "oneshot",
                "mode": mode,
                "state": "blocked",
                "backtest_completed": False,
            }
            write_json(tmp_path / "data/research/oneshot/status.json", status)
            write_json(tmp_path / "data/research/latest.json", status)
            return 0

    monkeypatch.setattr(
        "nifty_scalper_bot.ops.research_jobs.subprocess.Popen",
        lambda *args, **kwargs: Worker(),
    )
    result = start_job(tmp_path, {"id": "oneshot", "mode": mode}, wait=True)
    assert waited == [expected_timeout]
    assert result["state"] == "blocked"


@pytest.mark.parametrize("timeout", [True, False])
def test_updater_records_worker_timeout_or_unexpected_exit(
    tmp_path, monkeypatch, timeout
):
    import subprocess

    from nifty_scalper_bot.ops.research_jobs import start_job, write_json

    killed = []

    class Worker:
        def wait(self, timeout=None):
            if timeout is not None and globals_timeout:
                progress = {
                    "id": "dead-worker",
                    "mode": "all",
                    "state": "collecting",
                    "stage": "runtime_replay",
                    "backtest_completed": True,
                    "completed_trade_analysis": "completed_trade_analysis.json",
                }
                write_json(
                    tmp_path / "data/research/dead-worker/status.json",
                    progress,
                )
                write_json(tmp_path / "data/research/latest.json", progress)
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
    if timeout:
        assert result["timed_out_stage"] == "runtime_replay"
        assert result["worker_timeout_seconds"] == 3600
        assert result["completed_trade_analysis"] == "completed_trade_analysis.json"


def test_timeout_terminates_research_process_group(monkeypatch):
    import signal
    import subprocess

    from nifty_scalper_bot.ops.research_jobs import _terminate_worker

    signals = []

    class Worker:
        pid = 4321

        def wait(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired("research-worker", timeout)
            return 0

        def kill(self):
            raise AssertionError("process-group cleanup should be used")

    monkeypatch.setattr(
        "nifty_scalper_bot.ops.research_jobs.os.killpg",
        lambda pid, sig: signals.append((pid, sig)),
    )

    _terminate_worker(Worker())

    assert signals == [(4321, signal.SIGTERM), (4321, signal.SIGKILL)]


def test_worker_waits_for_startup_basket_and_redacts_unknown_errors(monkeypatch):
    import io

    from scripts import run_research_job as worker

    responses = iter(
        [OSError("secret auth detail"), {}, {"ce": "NFO:CE", "pe": "NFO:PE"}]
    )
    pauses = []

    def status(*args, **kwargs):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return io.StringIO(json.dumps({"selected": response}))

    monkeypatch.setattr(worker, "urlopen", status)
    monkeypatch.setattr(worker.time, "sleep", pauses.append)
    assert worker.load_active_basket()["ce"] == "NFO:CE"
    assert pauses == [2, 2]
    assert (
        worker.safe_error_code(ValueError("secret auth detail")) == "validation_failed"
    )
    assert (
        worker.safe_error_code(
            ValueError(
                "broker-calculated costs required for canonical research "
                "dataset: private-id"
            )
        )
        == "ledger_requires_verified_costs"
    )
    assert (
        worker.safe_error_code(
            ValueError(
                "Zerodha authentication invalid: Incorrect api_key or access_token"
            )
        )
        == "broker_authentication_invalid"
    )


def test_recorded_replay_budget_exhaustion_is_partial_not_completed(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from nifty_scalper_bot.ops.research_jobs import run_recorded_replays

    archive = tmp_path / "data/replay_archive"
    archive.mkdir(parents=True)
    for day in ("2026-09-30", "2026-10-01"):
        (archive / f"{day}.jsonl").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        "nifty_scalper_bot.config.paths.get_data_dir", lambda: tmp_path / "data"
    )
    clock = iter([0.0, 1.0, 1501.0])
    monkeypatch.setattr(
        "nifty_scalper_bot.ops.research_jobs.time.monotonic",
        lambda: next(clock),
    )

    def replay(command, **kwargs):
        output = Path(command[command.index("--output") + 1])
        (output / "report.json").write_text(
            json.dumps({"events_processed": 10}),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    from pathlib import Path

    monkeypatch.setattr(
        "nifty_scalper_bot.ops.research_jobs.subprocess.run",
        replay,
    )
    request = validate_request(
        {"id": "budgeted", "mode": "runtime", "days": 2},
        today=date(2026, 10, 2),
    )

    report = run_recorded_replays(tmp_path, request)

    assert report["state"] == "partial"
    assert report["budget_exhausted"] is True
    assert report["completed_sessions"] == 1
    assert report["failed_sessions"] == 0
    assert report["available_sessions_fully_processed"] is False
    assert report["deferred_from"] == "2026-10-01"


def test_runtime_mode_is_bounded_and_requires_no_operator_login(tmp_path, monkeypatch):
    from scripts import run_research_job as worker

    from nifty_scalper_bot.ops.research_jobs import run_recorded_replays

    monkeypatch.setattr(worker, "ROOT", tmp_path)
    monkeypatch.setattr(
        "nifty_scalper_bot.config.paths.get_data_dir", lambda: tmp_path / "data"
    )
    request = validate_request(
        {"id": "offline", "mode": "runtime", "days": 2}, today=date(2026, 10, 2)
    )
    result = worker.run_worker(request, None)
    assert result["state"] == "blocked"
    assert result["runtime_replay"]["blocker"] == "recorded_live_feed_unavailable"
    assert result["backtest_completed"] is False
    assert run_recorded_replays(tmp_path, request)["missing_dates"] == [
        "2026-09-30",
        "2026-10-01",
    ]
    with pytest.raises(ValueError):
        validate_request({"id": "bad", "mode": "shell"})

"""Date-pinned captured baseline research stays isolated and fails closed."""

import json
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from nifty_scalper_bot.ops.research_jobs import (
    _run_captured_baseline_stress,
    _same_request_definition,
    validate_request,
)

IST = ZoneInfo("Asia/Kolkata")


def test_date_pinned_baseline_requires_a_sealed_post_close_market_day():
    request = {
        "id": "baseline-20261104",
        "mode": "runtime",
        "days": 1,
        "profile": "captured_baseline",
        "session_date": "2026-11-04",
    }
    before = datetime(2026, 11, 4, 15, 59, tzinfo=IST)
    after = datetime(2026, 11, 4, 16, 0, tzinfo=IST)
    with pytest.raises(ValueError, match="post_close"):
        validate_request(request, now=before)
    parsed = validate_request(request, now=after)
    assert parsed["start"] == parsed["end"] == "2026-11-04"
    assert parsed["profile"] == "captured_baseline"
    assert _same_request_definition(parsed, parsed)

    for variant in (
        {"session_date": "2026-11-05"},
        {"session_date": "2026-02-01"},
        {"session_date": "../etc/"},
        {"mode": "all"},
        {"days": 2},
        {"profile": "unsafe_live"},
        {"command": "run live"},
    ):
        invalid = {**request, **variant}
        with pytest.raises(ValueError):
            validate_request(invalid, now=after)


def test_date_pinned_request_id_reuse_must_reject_different_session():
    a = validate_request(
        {"id": "one", "mode": "runtime", "days": 1, "session_date": "2026-11-03"},
        today=date(2026, 11, 5),
    )
    b = validate_request(
        {"id": "one", "mode": "runtime", "days": 1, "session_date": "2026-11-04"},
        today=date(2026, 11, 5),
    )
    assert not _same_request_definition(a, b)


def test_baseline_stresses_original_policy_and_checks_same_archive(
    tmp_path: Path, monkeypatch
):
    tape = tmp_path / "2026-11-04.jsonl"
    tape.write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(
        "nifty_scalper_bot.storage.replay_archive.archive_fingerprint",
        lambda _: "sha-contract",
    )
    calls = []

    def fake_worker(command, **kwargs):
        assert "--compare-adaptive" not in command
        assert kwargs["env"]["ALLOW_REAL_BROKER"] == "false"
        assert "ADAPTIVE_PROFIT_LOCK_ENABLED" not in kwargs["env"]
        assert command[command.index("--slippage-bps") + 1] in {"10", "25", "50"}
        output = Path(command[command.index("--output") + 1])
        output.mkdir(exist_ok=True, parents=True)
        output.joinpath("report.json").write_text(
            json.dumps(
                {
                    "archive_sha256": "sha-contract",
                    "captured_revision": "recorded-sha",
                    "slippage_bps_per_side": int(
                        command[command.index("--slippage-bps") + 1]
                    ),
                    "orders": [{"status": "COMPLETE"}],
                    "positions": [],
                    "modeled_account": {
                        "net_equity_pnl": -53,
                        "fees": 53,
                    },
                    "max_equity_drawdown": 53,
                }
            )
        )
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "nifty_scalper_bot.ops.research_jobs.subprocess.run", fake_worker
    )
    report = _run_captured_baseline_stress(
        tmp_path,
        tape,
        tmp_path / "output",
        {
            "ALLOW_REAL_BROKER": "false",
            "ENABLE_LIVE": "false",
            "ALLOW_NETWORK": "false",
        },
        Path("python"),
        deadline=__import__("time").monotonic() + 1000,
    )
    assert report["state"] == "completed"
    assert [row["slippage_bps_per_side"] for row in report["slippage_stress"]] == [
        10,
        25,
        50,
    ]
    assert len(calls) == 3
    assert report["promotion_eligible"] is False
    assert all(
        not row["exchange_stop_execution_proven"]
        for row in report["slippage_stress"]
    )


def test_baseline_rejects_unmatched_replay_sha(tmp_path: Path, monkeypatch):
    tape = tmp_path / "2026-11-04.jsonl"
    tape.write_text("fixture")
    monkeypatch.setattr(
        "nifty_scalper_bot.storage.replay_archive.archive_fingerprint",
        lambda _: "source-sha",
    )

    def bad_worker(command, **_kwargs):
        output = Path(command[command.index("--output") + 1])
        output.joinpath("report.json").write_text(
            json.dumps(
                {
                    "archive_sha256": "wrong-sha",
                    "slippage_bps_per_side": 10,
                    "orders": [],
                    "modeled_account": {"fees": 0},
                    "positions": [],
                }
            )
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "nifty_scalper_bot.ops.research_jobs.subprocess.run", bad_worker
    )
    report = _run_captured_baseline_stress(
        tmp_path,
        tape,
        tmp_path / "output",
        {},
        Path("python"),
        deadline=__import__("time").monotonic() + 1000,
    )
    assert report["state"] == "failed"
    assert report["slippage_stress"] == []
    assert report["error_code"] == "replay_baseline_scenario_failed"

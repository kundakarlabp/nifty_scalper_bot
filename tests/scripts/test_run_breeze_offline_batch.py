"""Research batch regression: immutable source lineage, safe resume and SHADOW-only."""

import json
import sqlite3
from pathlib import Path

import pytest
from scripts.run_breeze_offline_batch import run_batch
from scripts.stage_breeze_sqlite_research import run


def _staged(tmp_path: Path) -> tuple[dict[str, object], Path]:
    from tests.scripts.test_stage_breeze_sqlite_research import _sources

    breeze, eod, summary = _sources(tmp_path)
    stage = tmp_path / "staged"
    result = run(
        breeze,
        eod,
        stage,
        summary_path=summary,
        expected_windows=1,
        permit_partial=True,
    )
    assert result["staged_days"] == 1
    return {
        "breeze_db": breeze,
        "eod_db": eod,
        "summary_path": summary,
        "staged_dir": stage,
        "output_dir": tmp_path / "batch",
        "expected_windows": 1,
        "include_partial": True,
    }, stage


def test_resumed_runner_checks_code_config_and_data_digests(tmp_path, monkeypatch):
    args, _ = _staged(tmp_path)
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("ENABLE_LIVE", "false")
    monkeypatch.setenv("ALLOW_REAL_BROKER", "false")
    from nifty_scalper_bot.backtesting.strategy_research import load_archive

    invoked = []

    def fake_research(path: Path, *, slippage_bps: float = 10.0):
        invoked.append(path.name)
        _, _, digest = load_archive(path)
        strategies = {
            strategy: {
                "trades": [],
                "no_vote_reasons": {"no_setup": 1},
                "entry_funnel": {"evaluations": 1},
            }
            for strategy in ("ORBPro", "SMC", "VWAPPro")
        }
        return {
            "evidence_label": "RESEARCH_CANDIDATE",
            "live_equivalent": False,
            "data_sha256": digest,
            "scenarios": [
                {"slippage_bps_per_side": bps, "strategies": strategies}
                for bps in (10, 25, 50)
            ],
        }

    monkeypatch.setattr(
        "nifty_scalper_bot.backtesting.strategy_research.run_archived_research",
        fake_research,
    )
    first = run_batch(**args)
    assert first["eligible_sessions"] == first["newly_completed"] == 1
    assert first["live_promotion_eligible"] is False
    assert invoked == ["2025-02-04"]
    second = run_batch(**args)
    assert second["newly_completed"] == 0
    assert second["verified_resume_count"] == 1
    assert len(second["results"]) == 9
    assert invoked == ["2025-02-04"]
    path = args["output_dir"] / "runs/2025-02-04.json"
    item = json.loads(path.read_text())
    item["configuration_sha256"] = "0" * 64
    path.write_text(json.dumps(item))
    with pytest.raises(
        ValueError, match="research_resume_stale_or_mismatched_artifact"
    ):
        run_batch(**args)


def test_changes_to_eod_or_live_environment_fail_closed(tmp_path, monkeypatch):
    args, _ = _staged(tmp_path)
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    with pytest.raises(
        ValueError, match="breeze_offline_batch_requires_explicit_shadow"
    ):
        run_batch(**args)
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    with sqlite3.connect(args["eod_db"]) as db:
        db.execute("UPDATE contract_eod SET lot_size=25 WHERE kind='PE'")
    with pytest.raises(ValueError, match="staged_session_eod_digest_mismatch"):
        run_batch(**args)
    assert not args["output_dir"].exists()


def test_attempted_opening_minute_fill_is_never_accepted(tmp_path, monkeypatch):
    args, _ = _staged(tmp_path)
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("ENABLE_LIVE", "false")
    monkeypatch.setenv("ALLOW_REAL_BROKER", "false")
    from nifty_scalper_bot.backtesting.strategy_research import load_archive

    def fake_research(path: Path, *, slippage_bps: float = 10.0):
        _, _, digest = load_archive(path)
        return {
            "evidence_label": "RESEARCH_CANDIDATE",
            "live_equivalent": False,
            "data_sha256": digest,
            "scenarios": [
                {
                    "slippage_bps_per_side": 10.0,
                    "strategies": {
                        "SMC": {
                            "trades": [{"entry_time": "2025-02-04T09:15:00+05:30"}]
                        },
                    },
                }
            ],
        }

    monkeypatch.setattr(
        "nifty_scalper_bot.backtesting.strategy_research.run_archived_research",
        fake_research,
    )
    with pytest.raises(ValueError, match="research_trade_precedes_contract_selection"):
        run_batch(**args)
    assert not list((args["output_dir"] / "runs").glob("*.json"))

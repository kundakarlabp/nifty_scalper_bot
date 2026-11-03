"""Regression for full-depth WS markouts and reproducible session reporting."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

from scripts.reporting.rejected_setup_forward_markout import load_archive_quotes

from nifty_scalper_bot.infra.market_research_report import (
    _cost_model,
    _parity_summary,
    _sampled_markouts,
    build_session_research_report,
    research_report_log_lines,
)
from nifty_scalper_bot.storage.replay_archive import ReplayArchive

CONTRACT = "NFO:NIFTY26N0323700CE"
OTHER = "NFO:NIFTY26N0323700PE"
DECISION = datetime(2026, 11, 3, 6, 0, tzinfo=timezone.utc)


def _session(tmp_path):
    archive = ReplayArchive(tmp_path / "replay_archive")
    initial = DECISION - timedelta(seconds=2)
    archive.record("snapshot", {"basket": {"selected_ce": CONTRACT}}, initial)
    for sec in (-1, 1, 20, 40, 59, 90, 120, 150, 179, 210, 240, 270, 299):
        quote = {
            "symbol": CONTRACT,
            "source": "ws_full",
            "depth_available": True,
            "tradable_quote": True,
            "bid": 99.0 + sec / 10.0,
            "ask": 100.0 + sec / 10.0,
        }
        archive.record("tick", quote, DECISION + timedelta(seconds=sec))
    # Never let a REST quote contaminate the exchange-side markout.
    archive.record(
        "tick",
        {
            "symbol": CONTRACT,
            "source": "rest",
            "depth_available": True,
            "tradable_quote": True,
            "bid": 999,
            "ask": 1000,
        },
        DECISION + timedelta(seconds=300),
    )
    archive.close()
    return tmp_path / "replay_archive" / "2026-11-03.jsonl", archive.run_id


def _journal() -> str:
    fields = {
        "v": 1,
        "decision_at": DECISION.isoformat(),
        "evaluation_minute": DECISION.isoformat(),
        "strategy": "SMC",
        "symbol": CONTRACT,
        "reason": "underlying_no_liquidity_sweep",
    }
    parity = {
        "v": 1,
        "checked_at": DECISION.isoformat(),
        "indicator_minute": DECISION.isoformat(),
        "symbol": CONTRACT,
        "status": "same_minute_volume_match",
    }
    return (
        f"RESEARCH_STRATEGY_REJECTION {json.dumps(fields)}\n"
        f"OPTION_COMPLETED_VOLUME_PARITY {json.dumps(parity)}"
    )


def test_full_depth_ws_source_is_retained_but_rest_is_excluded(tmp_path) -> None:
    path, _run = _session(tmp_path)
    quotes, fingerprint, count = load_archive_quotes(path)
    assert len(fingerprint) == 64
    assert count == 13
    assert set(quotes) == {CONTRACT}
    assert max(row[1] for row in quotes[CONTRACT]) < 999


def test_report_uses_exact_contract_and_five_forward_horizons(tmp_path) -> None:
    path, _run = _session(tmp_path)
    destination = tmp_path / "research" / "markouts.json"
    report = build_session_research_report(
        session_date=date(2026, 11, 3),
        journal=_journal(),
        replay_path=path,
        trade_db=tmp_path / "not_present.db",
        output_path=destination,
    )
    assert report["status"] == "analysed"
    assert report["tradable_ws_quote_ticks"] == 13
    assert report["volume_parity"]["positive_volume_matches"] == 1
    assert [o["horizon_seconds"] for o in report["sampled_observations"]] == [
        60,
        180,
        300,
        600,
        900,
    ]
    assert all(o["markout"] for o in report["sampled_observations"][:3])
    assert all(o["markout"] is None for o in report["sampled_observations"][3:])
    assert all(
        o["cost_adjusted"] is None for o in report["sampled_observations"]
    )  # Sparse archived books must not support an MAE or net-P&L claim.
    assert all(
        o["independent_structural_setup"] is False
        for o in report["sampled_observations"]
    )
    assert report["eligible_for_entry_filter_change"] is False
    persisted = json.loads(destination.read_text())
    assert persisted["replay_sha256"] == report["replay_sha256"]
    lines = research_report_log_lines(report)
    assert lines[0].startswith("RESEARCH_SESSION_EVIDENCE ")
    assert any(line.startswith("RESEARCH_MARKOUT_STRATUM ") for line in lines)
    assert "canonical_trade_db_missing" == report["structural_data_blocker"]


def test_integrity_failure_never_claims_markouts(tmp_path) -> None:
    path, run_id = _session(tmp_path)
    sidecar = path.parent / f"{run_id}.integrity.json"
    evidence = json.loads(sidecar.read_text())
    evidence["dropped"] = 1
    sidecar.write_text(json.dumps(evidence))
    report = build_session_research_report(
        session_date=date(2026, 11, 3),
        journal=_journal(),
        replay_path=path,
        trade_db=tmp_path / "missing.db",
        output_path=tmp_path / "unavailable.json",
    )
    assert report["status"] == "unavailable"
    assert report["blocker"] == "replay_capture_incomplete"
    assert report["sampled_observations"] == []
    assert report["eligible_for_entry_filter_change"] is False


def test_zero_parity_samples_never_qualify_as_verified() -> None:
    rows = [
        {
            "symbol": CONTRACT,
            "indicator_minute": DECISION.isoformat(),
            "status": "zero_volume_match_unverified",
        },
        {
            "symbol": OTHER,
            "indicator_minute": DECISION.isoformat(),
            "status": "same_minute_volume_mismatch",
        },
    ]
    summary = _parity_summary(rows)
    assert summary["observations"] == 2
    assert summary["positive_volume_matches"] == 0
    assert summary["verified_fraction_of_observed"] == 0


def test_dense_replay_path_yields_five_cost_adjusted_horizons() -> None:
    import math

    at = DECISION.timestamp()
    quotes = {CONTRACT: [(at - 1.0, 100.0, 100.1)]}
    quotes[CONTRACT].extend(
        (at + second, 100.0 + 0.005 * second, 100.1 + 0.005 * second)
        for second in range(1, 901)
    )
    samples, strata = _sampled_markouts(
        [
            {
                "strategy": "SMC",
                "symbol": CONTRACT,
                "evaluation_minute": DECISION.isoformat(),
                "decision_at": DECISION.isoformat(),
                "reason": "underlying_no_liquidity_sweep",
                "context": {"selected_ce": CONTRACT, "vwap": 99.5},
            }
        ],
        quotes,
    )
    assert len(samples) == 5
    assert all(x["cost_adjusted"] is not None for x in samples)
    assert all(x["cost_adjusted"]["terminal"]["fees_rupees"] > 0 for x in samples)
    assert all(x["cost_adjusted"]["actual_stop_executed"] is None for x in samples)
    assert all(math.isfinite(x["cost_adjusted"]["terminal"]["net_rupees"]) for x in samples)
    assert all(g["cost_adjusted_labelled"] == 1 for g in strata)


def test_traded_contract_mismatch_blocks_counterfactual_markouts() -> None:
    at = DECISION.timestamp()
    rows = [(at - 1.0, 100.0, 100.1)]
    rows.extend((at + sec, 101.0, 101.1) for sec in range(1, 61))
    outcomes, _ = _sampled_markouts(
        [
            {
                "strategy": "VWAPPro",
                "symbol": CONTRACT,
                "evaluation_minute": DECISION.isoformat(),
                "decision_at": DECISION.isoformat(),
                "reason": "distance_outside_band",
                "context": {"selected_ce": OTHER},
            }
        ],
        {CONTRACT: rows},
    )
    assert all(row["markout"] is None for row in outcomes)
    assert all(row["label_status"] == "contract_not_selected_at_decision" for row in outcomes)


def test_spread_not_double_charged_for_ask_to_bid_quote_profit() -> None:
    modeled = _cost_model(
        {
            "entry_ask": 100.0,
            "terminal_bid": 102.0,
            "mae_markout_pct": -1.0,
            "mfe_markout_pct": 3.0,
        },
        quantity=65,
        adverse_slippage_bps=0,
    )
    assert modeled["terminal"]["gross_rupees"] == 130.0
    assert modeled["terminal"]["net_rupees"] < 130.0
    assert modeled["observed_mae"]["net_rupees"] < 0
    assert modeled["actual_stop_executed"] is None

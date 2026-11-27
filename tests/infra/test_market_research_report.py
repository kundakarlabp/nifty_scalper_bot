"""Regression for full-depth WS markouts and reproducible session reporting."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

from scripts.reporting.rejected_setup_forward_markout import load_archive_quotes

from nifty_scalper_bot.infra.market_research_report import (
    _cost_model,
    _parity_summary,
    _sampled_markouts,
    _session_decision_rows,
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


def test_parity_summary_preserves_lagged_common_minute_evidence() -> None:
    rows = [
        {
            "symbol": CONTRACT,
            "indicator_minute": DECISION.isoformat(),
            "status": "same_minute_volume_match",
            "latest_sources_aligned": False,
        },
        {
            "symbol": OTHER,
            "indicator_minute": DECISION.isoformat(),
            "status": "same_minute_volume_match",
            "latest_sources_aligned": True,
        },
    ]
    summary = _parity_summary(rows)
    assert summary["positive_volume_matches"] == 2
    assert summary["latest_source_unaligned_observations"] == 1
    assert summary["latest_aligned_positive_matches"] == 1


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
    assert all(
        math.isfinite(x["cost_adjusted"]["terminal"]["net_rupees"]) for x in samples
    )
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
    assert all(
        row["label_status"] == "contract_not_selected_at_decision" for row in outcomes
    )


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


def test_session_storage_flags_upstream_loss_and_reports_queue_lag(tmp_path):
    from nifty_scalper_bot.infra.market_research_report import capture_storage_status

    counters = {"dropped": 0, "coalesced": 0, "legacy_queue_dropped": 0}
    folder = tmp_path / "replay_archive"
    archive = ReplayArchive(folder, upstream_provider=lambda: dict(counters))
    archive.record("snapshot", {"basket": {}}, DECISION)
    archive.record("tick", {"symbol": CONTRACT, "ltp": 100}, DECISION)
    counters["coalesced"] = 1
    archive.close()
    quality = capture_storage_status(DECISION.date(), directory=folder)
    assert quality["upstream_lost_or_coalesced"] == 1
    assert quality["observed_loss_free"] is False
    assert "upstream_drop_or_coalescing" in quality["warnings"]
    assert quality["max_writer_queue_pending"] >= 1
    assert quality["max_writer_lag_ms"] >= 0


def test_postmarket_emits_a_persisted_fidelity_failure_without_fabricating_quotes(
    tmp_path, monkeypatch
):
    from nifty_scalper_bot.config import paths
    from nifty_scalper_bot.infra.market_research_report import (
        run_postmarket_research,
    )

    monkeypatch.setattr(paths, "get_data_dir", lambda: tmp_path)
    path, _ = _session(tmp_path)
    assert path.is_file()
    result = run_postmarket_research(DECISION.date(), _journal())
    assert result["status"] == "analysed"
    # A bare-basket unit fixture is not a full production snapshot; the
    # strict quality audit must preserve its blocker independently of markouts.
    assert result["capture_fidelity"]["status"] == "unavailable"
    report_path = tmp_path / "research" / "fidelity" / "2026-11-03.json"
    assert json.loads(report_path.read_text())["archive_integrity_verified"] is False
    text = "\n".join(research_report_log_lines(result))
    assert "REPLAY_SESSION_FIDELITY " in text


def test_postmarket_changed_archive_hash_invalidates_counterfactuals(
    tmp_path, monkeypatch
) -> None:
    from nifty_scalper_bot.backtesting import replay_fidelity
    from nifty_scalper_bot.config import paths
    from nifty_scalper_bot.infra.market_research_report import (
        run_postmarket_research,
    )

    monkeypatch.setattr(paths, "get_data_dir", lambda: tmp_path)
    path, _ = _session(tmp_path)
    assert path.is_file()
    monkeypatch.setattr(
        replay_fidelity,
        "audit_session",
        lambda *_args, **_kwargs: {"capture_sha256": "0" * 64},
    )
    report = run_postmarket_research(DECISION.date(), _journal())
    assert report["status"] == "unavailable"
    assert report["blocker"] == "replay_archive_changed_between_passes"
    assert report["sampled_observations"] == []
    assert report["structural_outcomes"] == []
    assert report["eligible_for_entry_filter_change"] is False
    assert report["capture_fidelity"]["archive_integrity_verified"] is False
    persisted = json.loads(
        (tmp_path / "research" / "markouts" / "2026-11-03.json").read_text()
    )
    assert persisted["status"] == "unavailable"


def test_session_decisions_are_scoped_by_ist_midnight_not_utc() -> None:
    midnight_ist = datetime(2026, 11, 2, 18, 30, tzinfo=timezone.utc)
    end_ist = midnight_ist + timedelta(days=1)
    rows = [
        {"timestamp": (midnight_ist - timedelta(seconds=1)).timestamp()},
        {"timestamp": midnight_ist.timestamp()},
        {"timestamp": DECISION.timestamp()},
        {"timestamp": (end_ist - timedelta(seconds=1)).timestamp()},
        {"timestamp": end_ist.timestamp()},
        {"timestamp": None},
        {"timestamp": float("nan")},
    ]
    scoped = _session_decision_rows(rows, date(2026, 11, 3))
    assert scoped == rows[1:4]


def test_postmarket_excludes_other_day_trade_decisions_before_markouts(
    tmp_path, monkeypatch
) -> None:
    from nifty_scalper_bot.backtesting import setup_opportunity_analysis
    from nifty_scalper_bot.journal import trade_ledger

    replay_path, _run_id = _session(tmp_path)
    trade_db = tmp_path / "trades.db"
    trade_db.touch()
    decisions = [
        {"timestamp": (DECISION - timedelta(days=1)).timestamp()},
        {"timestamp": DECISION.timestamp()},
        {"timestamp": (DECISION + timedelta(days=1)).timestamp()},
    ]
    seen = []

    monkeypatch.setattr(
        trade_ledger, "load_candidate_decision_rows", lambda _path: decisions
    )

    def capture(rows):
        seen.extend(rows)
        return ()

    monkeypatch.setattr(
        setup_opportunity_analysis, "canonicalize_setup_opportunities", capture
    )
    report = build_session_research_report(
        session_date=date(2026, 11, 3),
        journal=_journal(),
        replay_path=replay_path,
        trade_db=trade_db,
        output_path=tmp_path / "scope.json",
    )
    assert report["status"] == "analysed"
    assert seen == [decisions[1]]
    assert report["structural_decision_rows_scoped"] == 1
    assert report["structural_decision_rows_excluded_non_session"] == 2
    session_log = research_report_log_lines(report)[0]
    assert '"structural_decision_rows_scoped":1' in session_log
    assert '"structural_decision_rows_excluded_non_session":2' in session_log


def _range_excursion_session(tmp_path, *, bid_quantity: int = 65):
    future = "NFO:NIFTY26NOVFUT"
    archive = ReplayArchive(tmp_path / "replay_archive")
    archive.record(
        "snapshot",
        {
            "basket": {
                "selected_ce": CONTRACT,
                "selected_pe": OTHER,
                "futures_symbol": future,
                "token_by_symbol": {CONTRACT: 111, OTHER: 112, future: 113},
            }
        },
        DECISION - timedelta(seconds=2),
    )
    for second in [-1, *range(1, 901, 5)]:
        instant = DECISION + timedelta(seconds=second)
        archive.record(
            "tick",
            {
                "symbol": future,
                "instrument_token": 113,
                "source": "ws_full",
                "ltp": 25000.0 + 0.02 * second,
            },
            instant,
        )
        bid = 100.0 + 0.01 * second
        ask = bid + 0.5
        archive.record(
            "tick",
            {
                "symbol": CONTRACT,
                "instrument_token": 111,
                "source": "ws_full",
                "tradable_quote": True,
                "depth_available": True,
                "bid": bid,
                "ask": ask,
                "depth": {
                    "buy": [{"price": bid, "quantity": bid_quantity}],
                    "sell": [{"price": ask, "quantity": 75}],
                },
            },
            instant,
        )
    archive.close()
    first_bar = (DECISION - timedelta(minutes=1)).isoformat()
    identity = f"{future}:CE:{first_bar}"
    observation = {
        "v": 1,
        "decision_at": DECISION.isoformat(),
        "excursion_id": identity,
        "futures_symbol": future,
        "excursion_first_bar": first_bar,
        "side": "CE",
        "symbol": CONTRACT,
        "regime": "RANGE",
        "structural_qualified": False,
        "structural_reason": "session_rolling_center_disagree",
        "session_vwap": 25012.0,
        "rolling_center": 25030.0,
        "center_gap_atr": 1.5,
        "volatility_ratio": 1.2,
        "quote": {
            "bid": 99.99,
            "ask": 100.49,
            "lot_size": 65,
            "tradable": True,
        },
    }
    stage = {
        "v": 1,
        "decision_at": (DECISION + timedelta(milliseconds=150)).isoformat(),
        "excursion_id": identity,
        "reason": "session_rolling_center_disagree",
        "symbol": CONTRACT,
        "modeled_stop_risk": None,
        "modeled_net_rr": None,
    }
    journal = (
        f"RANGE_VWAP_EXCURSION_OBSERVED {json.dumps(observation)}\n"
        f"RANGE_VWAP_EXCURSION_DECISION {json.dumps(stage)}"
    )
    tape = tmp_path / "replay_archive" / "2026-11-03.jsonl"
    return tape, journal


def test_excursion_report_uses_full_lot_exact_token_books_and_futures(tmp_path) -> None:
    tape, journal = _range_excursion_session(tmp_path)
    report = build_session_research_report(
        session_date=date(2026, 11, 3),
        journal=journal,
        replay_path=tape,
        trade_db=tmp_path / "missing.db",
        output_path=tmp_path / "range_report.json",
    )
    assert report["status"] == "analysed"
    assert len(report["range_excursion_opportunities"]) == 1
    assert (
        report["cost_model_provenance"]["range_vwap_adverse_slippage_bps_per_side"]
        == 25.0
    )
    assert all(
        r["adverse_slippage_bps_per_side"] == 25.0
        for r in report["range_excursion_horizons"]
    )
    assert [r["horizon_seconds"] for r in report["range_excursion_horizons"]] == [
        300,
        600,
        900,
    ]
    assert all(
        row["observed_book_and_futures_complete"]
        for row in report["range_excursion_horizons"]
    )
    assert all(
        row["option_cost_adjusted"]["terminal"]["fees_rupees"] > 0
        for row in report["range_excursion_horizons"]
    )
    assert (
        report["range_excursion_horizons"][0]["futures_markout"][
            "signed_terminal_points"
        ]
        > 0
    )
    center = next(
        row
        for row in report["range_admission_comparisons"]
        if row["policy"] == "ignore_center_disagreement_only"
        and row["horizon_seconds"] == 300
    )
    assert center["unique_first_seen_excursions"] == 1
    assert center["full_lot_bid_futures_labelled"] == 1
    assert center["same_decision_cash_risk_unverified"] == 1
    assert center["verified_risk_and_observed_horizon"] == 0
    assert center["modeled_net_rupees_mean"] is None
    assert center["comparison_not_causal_or_promotable"] is True
    assert report["eligible_for_entry_filter_change"] is False
    assert json.loads((tmp_path / "range_report.json").read_text())[
        "range_excursion_opportunities"
    ]


def test_local_range_shadow_report_never_invents_risk(tmp_path) -> None:
    tape, journal = _range_excursion_session(tmp_path)
    observed, decision = journal.splitlines()
    prefix, payload = observed.split(" ", 1)
    row = json.loads(payload)
    row.update(
        regime="TREND",
        regime_raw="trend",
        regime_gate_allowed=True,
        local_range_candidate=True,
        local_range_session_vwap_setup=True,
        structural_qualified=True,
        structural_reason="qualified",
    )
    prefix_decision, payload_decision = decision.split(" ", 1)
    event = json.loads(payload_decision)
    event.update(
        reason="strategy_regime_ineligible",
        decision_at=row["decision_at"],
    )
    journal = f"{prefix} {json.dumps(row)}\n" f"{prefix_decision} {json.dumps(event)}"
    report = build_session_research_report(
        session_date=date(2026, 11, 3),
        journal=journal,
        replay_path=tape,
        trade_db=tmp_path / "missing.db",
        output_path=tmp_path / "local_range.json",
    )
    cohort = [
        row
        for row in report["range_admission_comparisons"]
        if row["policy"] == "shadow_local_range_in_trend"
    ]
    assert len(cohort) == 3
    assert all(row["unique_first_seen_excursions"] == 1 for row in cohort)
    assert all(row["verified_risk_and_observed_horizon"] == 0 for row in cohort)
    assert all(row["modeled_net_rupees_mean"] is None for row in cohort)
    assert all(row["observed_full_lot_net_rupees_mean"] is None for row in cohort)
    assert any(
        '"kind":"range_admission"' in line
        and '"policy":"shadow_local_range_in_trend"' in line
        for line in research_report_log_lines(report)
    )
    assert report["eligible_for_entry_filter_change"] is False


def test_local_regime_clock_reference_is_prior_only_and_deduplicated(tmp_path) -> None:
    from nifty_scalper_bot.infra.market_research_report import _local_regime_minutes

    day = DECISION.date()
    stamp = DECISION - timedelta(minutes=1)
    sample = {
        "v": 1,
        "decision_at": DECISION.isoformat(),
        "bar_timestamp": stamp.isoformat(),
        "futures_symbol": "NFO:NIFTY26NOVFUT",
        "source": "contiguous_session_futures_1m",
        "futures_atr_bps": 4.0,
        "local_range_state": "RANGE",
        "local_reclaim_side": "CE",
        "regime": "trend",
        "regime_raw": "trend",
        "regime_gate_allowed": True,
        "session_vwap_setup_qualified": False,
    }
    for offset in (-5, -4, -3, -2, -1, 0, 1):
        prior_date = day + timedelta(days=offset)
        reference = dict(
            sample,
            bar_timestamp=(stamp + timedelta(days=offset)).isoformat(),
            futures_atr_bps=2.0 if offset < 0 else 100.0,
        )
        (tmp_path / f"{prior_date}.json").write_text(
            json.dumps(
                {"date_ist": str(prior_date), "local_regime_minutes": [reference]}
            )
        )
    rows, summary = _local_regime_minutes(
        [sample, dict(sample, futures_atr_bps=99.0)],
        session_date=day,
        report_directory=tmp_path,
    )
    assert len(rows) == 1
    assert rows[0]["clock_matched_prior_sessions"] == 5
    assert rows[0]["atr_vs_clock_matched_median"] == 2.0
    assert summary["confirmed_local_range_in_trend_minutes"] == 1
    assert summary["local_reclaim_without_session_setup_minutes"] == 1
    denied = dict(
        sample,
        regime="unknown",
        regime_raw="unknown",
        regime_gate_allowed=False,
        shadow_transition_candidate="RANGE",
    )
    _, summary = _local_regime_minutes(
        [denied], session_date=day, report_directory=tmp_path
    )
    assert summary["canonical_denied_observed_minutes"] == 1
    assert summary["unknown_with_confirmed_local_range_minutes"] == 1
    assert summary["shadow_transition_counts"] == {"RANGE": 1}
    assert summary["clock_matched_calibration_status"] == (
        "reference_available_research_only"
    )


def test_excursion_reports_missing_full_lot_depth_without_invented_exit(
    tmp_path,
) -> None:
    tape, journal = _range_excursion_session(tmp_path, bid_quantity=64)
    report = build_session_research_report(
        session_date=date(2026, 11, 3),
        journal=journal,
        replay_path=tape,
        trade_db=tmp_path / "missing.db",
        output_path=tmp_path / "missing_depth_report.json",
    )
    assert report["status"] == "analysed"
    assert all(
        row["futures_markout"] is not None
        and row["option_full_lot_markout"] is None
        and row["option_cost_adjusted"] is None
        for row in report["range_excursion_horizons"]
    )
    assert all(
        row["verified_risk_and_observed_horizon"] == 0
        for row in report["range_admission_comparisons"]
    )


def test_sampled_markouts_keep_regime_and_gate_evidence_separate() -> None:
    samples = []
    for offset, context in enumerate(
        (
            {"regime": "trend", "regime_raw": "TREND", "regime_gate_allowed": True},
            {"regime": "trend", "regime_raw": "unknown", "regime_gate_allowed": False},
            {},
        )
    ):
        instant = DECISION + timedelta(minutes=offset)
        samples.append(
            {
                "strategy": "SMC",
                "symbol": CONTRACT,
                "decision_at": instant.isoformat(),
                "evaluation_minute": instant.isoformat(),
                "reason": "strategy_regime_ineligible",
                "context": context,
            }
        )
    _, strata = _sampled_markouts(samples, {})
    assert {
        (row["regime"], row["regime_raw"], row["regime_gate_state"])
        for row in strata
        if row["horizon_seconds"] == 300
    } == {
        ("TREND", "TREND", "allowed"),
        ("TREND", "UNKNOWN", "blocked"),
        ("UNRECORDED", "UNRECORDED", "unrecorded"),
    }
    assert all(row["sampled_minutes"] == 1 for row in strata)
    assert all(row["eligible_for_filter_change"] is False for row in strata)


def test_unknown_local_reclaim_requires_separate_session_confirmation(tmp_path) -> None:
    tape, journal = _range_excursion_session(tmp_path)
    observed, decision = journal.splitlines()
    prefix, payload = observed.split(" ", 1)
    row = json.loads(payload)
    row.update(
        regime="UNKNOWN",
        regime_raw="unknown",
        regime_gate_allowed=False,
        shadow_transition_candidate="RANGE",
        local_range_state="RANGE",
        local_reclaim_side="CE",
        structural_qualified=False,
    )
    decision_prefix, decision_payload = decision.split(" ", 1)
    event = json.loads(decision_payload)
    event.update(
        reason="canonical_regime_gate_block",
        decision_at=row["decision_at"],
    )
    for qualified in (False, True):
        row["structural_qualified"] = qualified
        journal = (
            f"{prefix} {json.dumps(row)}\n" f"{decision_prefix} {json.dumps(event)}"
        )
        report = build_session_research_report(
            session_date=date(2026, 11, 3),
            journal=journal,
            replay_path=tape,
            trade_db=tmp_path / "missing.db",
            output_path=tmp_path / "unknown_range.json",
        )
        groups = {
            item["policy"]: item
            for item in report["range_admission_comparisons"]
            if item["horizon_seconds"] == 300
        }
        assert (
            groups["shadow_unknown_local_range"]["unique_first_seen_excursions"] == 1
        )
        strict = groups["shadow_unknown_session_reclaim"]
        assert strict["unique_first_seen_excursions"] == int(qualified)
        assert strict["verified_risk_and_observed_horizon"] == 0
        assert strict["modeled_net_rupees_mean"] is None
        assert report["eligible_for_entry_filter_change"] is False


def test_rolling_vwap_reclaim_is_separate_and_never_inherits_session_risk(
    tmp_path,
) -> None:
    tape, journal = _range_excursion_session(tmp_path)
    observed, _ = journal.splitlines()
    prefix, payload = observed.split(" ", 1)
    row = json.loads(payload)
    row.update(
        excursion_basis="rolling_vwap",
        excursion_id=(
            f"{row['futures_symbol']}:{row['side']}:LOCAL:{row['excursion_first_bar']}"
        ),
        local_range_state="RANGE",
        local_reclaim_side="CE",
        structural_reason="rolling_vwap_reclaim_shadow_only",
        structural_qualified=False,
    )
    forged_risk = {
        "v": 1,
        "decision_at": row["decision_at"],
        "excursion_id": row["excursion_id"],
        "symbol": CONTRACT,
        "reason": "signal_emitted",
        "modeled_stop_risk": 500,
        "modeled_net_rr": 2.0,
    }
    journal += (
        f"\n{prefix} {json.dumps(row)}\n"
        f"RANGE_VWAP_EXCURSION_DECISION {json.dumps(forged_risk)}"
    )
    report = build_session_research_report(
        session_date=date(2026, 11, 3),
        journal=journal,
        replay_path=tape,
        trade_db=tmp_path / "missing.db",
        output_path=tmp_path / "rolling_only.json",
    )
    assert len(report["range_excursion_opportunities"]) == 2
    local = [
        item
        for item in report["range_excursion_horizons"]
        if item["excursion_basis"] == "rolling_vwap"
    ]
    assert len(local) == 3
    assert all(item["observed_book_and_futures_complete"] for item in local)
    assert all(not item["same_decision_risk_773_and_net_rr_verified"] for item in local)
    assert all(not item["baseline_signal_emitted_at_first_decision"] for item in local)
    assert report["eligible_for_entry_filter_change"] is False

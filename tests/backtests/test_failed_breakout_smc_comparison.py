"""Causal same-event SMC/reversal comparisons never authorize a live strategy."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.backtesting import failed_breakout_smc_comparison as audit

IST = timezone(timedelta(hours=5, minutes=30))
START = datetime(2026, 10, 1, 10, 0, tzinfo=IST)
FUTURE = "NFO:NIFTY26OCTFUT"
PE = "NFO:NIFTY26O0624000PE"


def _reversal_bars() -> list[dict]:
    rows = [
        {
            "timestamp": START + timedelta(minutes=i),
            "open": 24000.0,
            "high": 24001.0,
            "low": 23999.0,
            "close": 24000.0,
            "volume": 1000.0,
            "is_complete": True,
            "is_provisional": False,
        }
        for i in range(45)
    ]
    rows[-2] = {
        **rows[-2],
        "high": 24005.0,
        "low": 23999.0,
        "volume": 1600.0,
    }
    rows[-1] = {
        **rows[-1],
        "open": 24000.0,
        "high": 24000.5,
        "low": 23997.0,
        "close": 23998.0,
        "volume": 1200.0,
    }
    return rows


def _archive_parts():
    bars = _reversal_bars()
    futures = {row["timestamp"]: row for row in bars}
    last = bars[-1]["timestamp"]
    return (
        {
            FUTURE: futures,
            "NSE:NIFTY 50": {last: {"close": 24000.0}},
            PE: {last: {"volume": 1200.0}},
        },
        {
            FUTURE: {"instrument_type": "FUT"},
            PE: {
                "instrument_type": "PE",
                "strike": 24000,
                "lot_size": 65,
                "expiry": "2026-10-06",
            },
        },
    )


def test_common_reversal_event_cohort_is_confirmed_and_has_no_next_bar_lookahead():
    histories, instruments = _archive_parts()
    rows = audit.completed_reversal_event_cohort(histories, instruments)
    assert len(rows) == 1
    event = rows[0]
    assert event["side"] == "PE"
    assert event["symbol"] == PE
    assert event["event_bar_start"] == (
        START + timedelta(minutes=44)
    ).isoformat()
    assert event["earliest_entry"] == (
        START + timedelta(minutes=45)
    ).isoformat()
    # The next minute is absent from every source. Eligibility is still causal.
    assert START + timedelta(minutes=45) not in histories[FUTURE]
    assert START + timedelta(minutes=45) not in histories[PE]

    unconfirmed, _ = _archive_parts()
    unconfirmed[FUTURE][START + timedelta(minutes=44)] = {
        **unconfirmed[FUTURE][START + timedelta(minutes=44)],
        "is_complete": False,
    }
    assert audit.completed_reversal_event_cohort(unconfirmed, instruments) == []


def test_exact_next_minute_contract_matching_preserves_abstentions_and_costs():
    histories, instruments = _archive_parts()
    cohort = audit.completed_reversal_event_cohort(histories, instruments)
    entry = cohort[0]["earliest_entry"]
    trade = {
        "entry_time": entry,
        "symbol": PE,
        "cost_source": "canonical_model_estimate",
        "net_pnl": 83.0,
        "fees": 42.0,
    }
    scenario = {
        "slippage_bps_per_side": 10.0,
        "strategies": {
            "SMC": {"trades": [], "unresolved_trades": []},
            "FailedBreakoutResearch": {
                "trades": [trade],
                "unresolved_trades": [],
            },
        },
    }
    compared = audit.compare_smc_reversal_events(cohort, scenario)
    assert compared["underlying_event_count"] == 1
    assert compared["matched_option_event_count"] == 1
    assert compared["resolved_fills"] == {"SMC": 0, "FailedBreakoutResearch": 1}
    assert compared["reversal_only_positive_resolved_fill_count"] == 1
    assert compared["outperformance_inference_allowed"] is False
    smc = compared["events"][0]["outcomes"]["SMC"]
    assert smc["absence_means_no_signal"] is False
    assert smc["net_pnl"] is None

    late = {**trade, "entry_time": (datetime.fromisoformat(entry) + timedelta(minutes=1)).isoformat()}
    changed = {
        **scenario,
        "strategies": {
            **scenario["strategies"],
            "FailedBreakoutResearch": {
                "trades": [late],
                "unresolved_trades": [],
            },
        },
    }
    assert audit.compare_smc_reversal_events(cohort, changed)["resolved_fills"][
        "FailedBreakoutResearch"
    ] == 0


def test_unresolved_exit_and_duplicate_trade_do_not_become_profit():
    cohort = [
        {
            "event_bar_start": START.isoformat(),
            "earliest_entry": (START + timedelta(minutes=1)).isoformat(),
            "symbol": PE,
            "side": "PE",
        }
    ]
    entry = cohort[0]["earliest_entry"]
    unresolved = {"entry_time": entry, "symbol": PE}
    scenario = {
        "slippage_bps_per_side": 25.0,
        "strategies": {
            "SMC": {"trades": [], "unresolved_trades": []},
            "FailedBreakoutResearch": {
                "trades": [],
                "unresolved_trades": [unresolved],
            },
        },
    }
    compared = audit.compare_smc_reversal_events(cohort, scenario)
    assert compared["unresolved_exits"]["FailedBreakoutResearch"] == 1
    assert compared["reversal_only_positive_resolved_fill_count"] == 0
    changed = {
        **scenario,
        "strategies": {
            **scenario["strategies"],
            "FailedBreakoutResearch": {
                "trades": [],
                "unresolved_trades": [unresolved, unresolved],
            },
        },
    }
    with pytest.raises(ValueError, match="reversal_matched_duplicate_trade"):
        audit.compare_smc_reversal_events(cohort, changed)


def test_comparison_uses_same_canonical_replay_not_a_second_execution_engine(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    histories, instruments = _archive_parts()
    monkeypatch.setattr(
        audit,
        "load_archive",
        lambda _dir: (histories, instruments, "verified_archive_digest"),
    )
    monkeypatch.setattr(
        audit,
        "get_settings",
        lambda: SimpleNamespace(elite=SimpleNamespace(smc=object())),
    )
    invocations = []

    def fake_scenario(_histories, _instruments, _settings, slip, **kwargs):
        invocations.append(
            {
                "slip": slip,
                "components": kwargs["components"],
                "strict_liquidity": kwargs["strict_liquidity"],
                "minimum_net_rr": kwargs["minimum_net_rr"],
                "weekly": kwargs["require_archived_weekly_atm"],
            }
        )
        return {
            "slippage_bps_per_side": slip,
            "strategies": {
                "SMC": {"trades": [], "unresolved_trades": []},
                "FailedBreakoutResearch": {"trades": [], "unresolved_trades": []},
            },
        }

    monkeypatch.setattr(audit, "_scenario", fake_scenario)
    result = audit.run_failed_breakout_smc_comparison(tmp_path)
    assert [r["slip"] for r in invocations] == [10.0, 25.0, 50.0]
    assert all(
        r["components"] == {"SMC", "FailedBreakoutResearch"}
        and r["strict_liquidity"]
        and r["minimum_net_rr"] == 1.5
        and r["weekly"]
        for r in invocations
    )
    assert result["promotion_eligible"] is False
    assert result["live_equivalent"] is False
    assert result["scenarios"][0]["underlying_event_count"] == 1

    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    with pytest.raises(ValueError, match="reversal_smc_comparison_requires_shadow"):
        audit.run_failed_breakout_smc_comparison(tmp_path)

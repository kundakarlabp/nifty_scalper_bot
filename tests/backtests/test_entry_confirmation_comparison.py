"""Causal, no-live-promotion regressions for three-strategy entry ablation."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from nifty_scalper_bot.backtesting import strategy_research as research


def _bar(ts: datetime, *, opening: float, closing: float, volume: float = 10) -> dict:
    return {
        "timestamp": ts,
        "open": opening,
        "close": closing,
        "volume": volume,
        "is_complete": True,
        "is_provisional": False,
    }


def test_completed_option_confirmation_is_causal_and_requires_contiguous_trading():
    now = datetime(2026, 10, 14, 10, 1, tzinfo=ZoneInfo("Asia/Kolkata"))
    previous = _bar(now - timedelta(minutes=1), opening=100, closing=101)
    assert (
        research._option_premium_confirmation_reason(
            _bar(now, opening=101, closing=104), previous
        )
        is None
    )
    assert (
        research._option_premium_confirmation_reason(
            _bar(now, opening=101, closing=100), previous
        )
        == "option_premium_not_confirmed"
    )
    assert (
        research._option_premium_confirmation_reason(
            _bar(now, opening=101, closing=104),
            _bar(now - timedelta(minutes=2), opening=100, closing=101),
        )
        == "option_confirmation_history_unavailable"
    )
    assert (
        research._option_premium_confirmation_reason(
            _bar(now, opening=101, closing=104, volume=0), previous
        )
        == "option_confirmation_history_unavailable"
    )


@dataclass
class _Config:
    some_setting: float = 1.0


def test_entry_comparison_uses_paired_all_three_strategy_arms(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    settings = SimpleNamespace(
        elite=SimpleNamespace(orb=_Config(), smc=_Config(), vwap=_Config())
    )
    monkeypatch.setattr(research, "get_settings", lambda: settings)
    monkeypatch.setattr(research, "load_archive", lambda _: ({}, {}, "archive_sha"))
    observed = []

    def fake_scenario(histories, instruments, elite, slippage, **constraints):
        observed.append((slippage, constraints))
        assert constraints["components"] == {"ORBPro", "SMC", "VWAPPro"}
        return {
            "strategies": {
                name: {
                    "metrics": {"trade_count": 0, "net_pnl": 0},
                    "evaluations": 0,
                    "data_quality": {
                        "eligible_option_bar_count": (
                            0 if constraints.get("require_archived_weekly_atm") else None
                        ),
                        "eligible_session_count": 0,
                    },
                    "entry_funnel": {"signals_emitted": 0},
                    "no_vote_reasons": {"no_signal": 1},
                    "trades": [{"test": True}],
                    "unresolved_trades": [],
                    "worst_case_stress_trades": [],
                }
                for name in ("ORBPro", "SMC", "VWAPPro")
            }
        }

    monkeypatch.setattr(research, "_scenario", fake_scenario)
    report = research.run_entry_confirmation_comparison(Path("/unused"))
    assert report["candidate_count"] == 3
    assert report["data_sha256"] == "archive_sha"
    assert len(observed) == 9
    assert [row[0] for row in observed] == [10, 25, 50] * 3
    assert "minimum_net_rr" not in observed[0][1]
    assert observed[3][1]["minimum_net_rr"] == 1.5
    assert observed[3][1]["strict_liquidity"] is True
    assert observed[3][1]["require_archived_weekly_atm"] is True
    assert observed[6][1]["require_option_premium_confirmation"] is True
    assert report["selection"]["selected_for_live"] is None
    assert report["selection"]["promotion_eligible"] is False
    assert report["historical_eligibility"]["status"] == "not_evaluable"
    assert all(
        "trades" not in results
        for arm in report["candidates"]
        for scenario in arm["scenarios"]
        for results in scenario["strategies"].values()
    )


def test_entry_comparison_fails_closed_in_live_mode(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    with pytest.raises(ValueError, match="research_requires_shadow_process"):
        research.run_entry_confirmation_comparison(Path("/unused"))


def test_zero_eligible_option_bars_are_not_misreported_as_no_signal():
    result = research._historical_weekly_atm_evaluability(
        {
            "evaluations": 0,
            "data_quality": {
                "source_option_session_count": 19,
                "eligible_session_count": 0,
                "eligible_session_dates": [],
                "eligible_option_bar_count": 0,
            },
            "no_vote_reasons": {
                "archive_weekly_expiry_not_proven": 31033,
                "archive_strike_outside_atm_basket": 7896,
            },
        }
    )
    assert result["status"] == "not_evaluable"
    assert result["reason_code"] == "INSUFFICIENT_HISTORICAL_WEEKLY_ATM_COVERAGE"
    assert result["weekly_expiry_rejections"] == 31033
    assert result["outside_atm_rejections"] == 7896
    assert result["eligible_session_count"] == 0
    assert result["historical_selection_verified"] is False

    evaluated = research._historical_weekly_atm_evaluability(
        {
            "evaluations": 12,
            "data_quality": {
                "eligible_option_bar_count": 12,
                "eligible_session_count": 2,
                "eligible_session_dates": ["2026-10-01", "2026-10-02"],
            },
        }
    )
    assert evaluated["status"] == "research_proxy_only"
    assert evaluated["reason_code"] is None
    assert evaluated["eligible_session_dates"] == ["2026-10-01", "2026-10-02"]


def test_vwap_strict_comparison_reports_archive_ineligibility(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setattr(
        research,
        "get_settings",
        lambda: SimpleNamespace(elite=SimpleNamespace(vwap=_Config())),
    )
    monkeypatch.setattr(research, "load_archive", lambda _: ({}, {}, "archive_sha"))

    def ineligible_scenario(*_args, **_kwargs):
        return {
            "strategies": {
                "VWAPPro": {
                    "evaluations": 0,
                    "data_quality": {
                        "eligible_option_bar_count": 0,
                        "eligible_session_count": 0,
                        "source_option_session_count": 19,
                        "development_unresolved_exit_count": 0,
                    },
                    "no_vote_reasons": {
                        "archive_weekly_expiry_not_proven": 31033,
                        "archive_strike_outside_atm_basket": 7896,
                    },
                    "metrics": {"trade_count": 0, "expectancy": None},
                    "development_metrics": {"trade_count": 0, "expectancy": None},
                }
            }
        }

    monkeypatch.setattr(research, "_scenario", ineligible_scenario)
    report = research.run_vwap_comparison(Path("/unused"))
    assert report["candidate_count"] == 9
    assert report["historical_eligibility"]["status"] == "not_evaluable"
    assert report["historical_eligibility"]["eligible_option_bar_count"] == 0
    assert report["selection"]["promotion_eligible"] is False
    assert report["selection"]["selected_for_live"] is None

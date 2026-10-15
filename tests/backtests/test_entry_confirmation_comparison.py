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

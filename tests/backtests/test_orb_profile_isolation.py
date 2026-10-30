"""Research ORB comparisons must use isolated, reproducible scenario settings."""

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from nifty_scalper_bot.backtesting import strategy_research as research


@dataclass
class _Orb:
    orb_minutes: int = 15


@dataclass
class _Elite:
    orb: _Orb = field(default_factory=_Orb)


def test_orb_research_resets_inherited_toggles_and_restores_environment(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("ORB_MOMENTUM_BRANCH_ENABLED", "false")
    monkeypatch.setenv("ORB_MAX_ENTRY_MINUTES_AFTER_RANGE", "5")
    monkeypatch.setenv("ORB_RESEARCH_ALLOW_RANGE_TRANSITION", "true")
    monkeypatch.setenv("ORB_RESEARCH_STRICT_RETEST_WICK", "true")
    monkeypatch.setenv("ORB_TARGET_RR", "3.0")
    monkeypatch.setattr(
        research, "get_settings", lambda: SimpleNamespace(elite=_Elite())
    )
    monkeypatch.setattr(research, "load_archive", lambda _: ({}, {}, "sha"))
    seen = []

    def fake_scenario(*args, **kwargs):
        seen.append(
            {
                "slippage": args[3],
                "minutes": args[2].orb.orb_minutes,
                "momentum": research.os.getenv("ORB_MOMENTUM_BRANCH_ENABLED"),
                "entry_window": research.os.getenv("ORB_MAX_ENTRY_MINUTES_AFTER_RANGE"),
                "target_rr": research.os.getenv("ORB_TARGET_RR"),
                "allow_range": research.os.getenv(
                    "ORB_RESEARCH_ALLOW_RANGE_TRANSITION"
                ),
                "strict_wick": research.os.getenv("ORB_RESEARCH_STRICT_RETEST_WICK"),
            }
        )
        return {
            "strategies": {
                "ORBPro": {
                    "development_metrics": {"trade_count": 0, "expectancy": None},
                    "data_quality": {"development_unresolved_exit_count": 0},
                }
            }
        }

    monkeypatch.setattr(research, "_scenario", fake_scenario)
    result = research.run_orb_comparison(Path("/unused"))
    assert result["candidate_count"] == 11
    assert result["selection"]["promotion_eligible"] is False
    assert len(seen) == 11 * 3
    assert [entry["slippage"] for entry in seen[:3]] == [10.0, 25.0, 50.0]

    # Every reference starts from declared defaults, not operator leftovers.
    for entry in seen[:6]:
        assert entry["momentum"] is None
        assert entry["entry_window"] is None
        assert entry["target_rr"] is None
        assert entry["allow_range"] == "false"
        assert entry["strict_wick"] == "false"

    # The dedicated retest-only arm changes just its own momentum toggle.
    for entry in seen[9:12]:
        assert entry["momentum"] == "false"
        assert entry["entry_window"] is None
        assert entry["target_rr"] is None

    # Each scenario restores external settings without side effects.
    assert research.os.environ["ORB_MOMENTUM_BRANCH_ENABLED"] == "false"
    assert research.os.environ["ORB_MAX_ENTRY_MINUTES_AFTER_RANGE"] == "5"
    assert research.os.environ["ORB_RESEARCH_ALLOW_RANGE_TRANSITION"] == "true"
    assert research.os.environ["ORB_RESEARCH_STRICT_RETEST_WICK"] == "true"
    assert research.os.environ["ORB_TARGET_RR"] == "3.0"

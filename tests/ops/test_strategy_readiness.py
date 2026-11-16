from __future__ import annotations

from nifty_scalper_bot.ops.strategy_readiness import build_strategy_readiness


def test_strategy_readiness_distinguishes_arming_from_strategy_readiness(monkeypatch):
    monkeypatch.setenv("VWAP_EMA_PERIOD", "50")
    runtime = {
        "state": {"live_orders_armed": True},
        "selected": {"ce": "NFO:NIFTY26N1725250CE", "pe": "NFO:NIFTY26N1725250PE"},
        "history": {
            "ce": {"mdm": 44, "runner": 41, "indicator": 41},
            "pe": {"mdm": 60, "runner": 60, "indicator": 60},
        },
    }
    signals = {
        "no_vote_reason_counts_by_strategy_window": {
            "ORBPro": {
                "no_fresh_underlying_breakout": 1,
                "orb_bar_already_evaluated": 12,
            },
            "SMC": {"smc_awaiting_confirmation": 8},
        },
        "setup_lifecycle": {"active": [{"strategy": "SMC", "stage": "CONFIRMING"}]},
    }
    result = build_strategy_readiness(runtime, signals)
    assert result["VWAPPro"]["state"] == "WARMING"
    assert result["VWAPPro"]["ce"]["usable_bars"] == 41
    assert result["VWAPPro"]["ce"]["required_bars"] == 50
    assert result["VWAPPro"]["pe"]["history_ready"] is True
    assert result["SMC"]["state"] == "AWAITING_CONFIRMATION"
    assert result["ORBPro"]["state"] == "RANGE_EVALUATED_RECENTLY"


def test_strategy_readiness_missing_evidence_never_claims_ready():
    result = build_strategy_readiness({}, {})
    assert result["VWAPPro"]["state"] == "UNKNOWN"
    assert result["ORBPro"]["state"] == "UNKNOWN"
    assert result["SMC"]["state"] == "UNKNOWN"

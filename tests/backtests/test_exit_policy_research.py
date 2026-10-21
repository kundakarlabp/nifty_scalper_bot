"""Regression tests for research-only executable-bid exit overlays."""

from __future__ import annotations

import pytest

from nifty_scalper_bot.backtesting import exit_policy_research as research
from nifty_scalper_bot.storage.replay_archive import ReplayArchive


ENTRY = "2026-10-21T04:03:00+00:00"
SYMBOL = "NFO:NIFTY26OCT23100CE"


def trade() -> dict:
    return {
        "trade_id": "research-entry",
        "symbol": SYMBOL,
        "side": "BUY",
        "quantity": 65,
        "entry_price": 100,
        "initial_stop": 90,
        "net_pnl": 200,
        "effective_costs": 70,
        "ledger_complete": True,
        "entry_filled_at": ENTRY,
        "exit_filled_at": "2026-10-21T04:03:06+00:00",
    }


def tick(second: int, bid: float, *, qty: int = 65, symbol: str = SYMBOL) -> dict:
    return {
        "kind": "tick",
        "available_at": f"2026-10-21T04:03:{second:02d}+00:00",
        "payload": {
            "symbol": symbol,
            "ltp": bid + 1.0,
            "depth": {"buy": [{"price": bid, "quantity": qty}]},
        },
    }


def test_target_overlays_use_causal_visible_bid_and_costs() -> None:
    data = [tick(1, 101), tick(2, 106), tick(3, 108.2), tick(4, 110.2),
            tick(5, 109), tick(6, 107)]
    report = research.compare_trade(trade(), data)
    assert report["coverage_ready"]
    assert report["outcomes"]["target_0.6r"]["status"] == "MODELED_TARGET"
    assert report["outcomes"]["target_0.6r"]["exit_price"] == 105.5
    assert report["outcomes"]["target_0.6r"]["net"] == 287.5
    assert report["outcomes"]["target_0.8r"]["net"] == 417.5
    assert report["outcomes"]["target_1.0r"]["net"] == 547.5
    assert report["evidence"].startswith("RESEARCH_CANDIDATE")


def test_lock_fractions_act_after_peak_and_bounded_by_real_exit() -> None:
    data = [tick(1, 101), tick(2, 105), tick(3, 111), tick(4, 108),
            tick(5, 105), tick(6, 104)]
    report = research.compare_trade(trade(), data)
    assert report["outcomes"]["lock_40pct"]["status"] == "MODELED_STOP"
    assert report["outcomes"]["lock_50pct"]["status"] == "MODELED_STOP"
    assert report["outcomes"]["lock_60pct"]["status"] == "MODELED_STOP"
    assert report["outcomes"]["lock_60pct"]["at"].endswith("04:03:05+00:00")


def test_ltp_without_observed_executable_bid_is_never_used() -> None:
    data = [tick(1, 120, qty=1), tick(2, 120, qty=0), tick(3, 110, qty=1)]
    report = research.compare_trade(trade(), data)
    assert report["coverage_ready"] is False
    assert all(
        value["status"] == "UNRESOLVED"
        for value in report["outcomes"].values()
    )


def test_missing_quotes_reject_fake_profit_from_unobserved_gap() -> None:
    data = [tick(1, 101), tick(2, 102), tick(30, 115)]
    report = research.compare_trade(trade(), data)
    assert not report["coverage_ready"]
    assert report["outcomes"]["target_0.6r"]["status"] == "UNRESOLVED"


def test_future_quote_outside_actual_holding_window_cannot_satisfy_target() -> None:
    data = [tick(1, 100), tick(2, 101), tick(3, 102), tick(4, 103),
            tick(5, 102), tick(6, 101), tick(7, 120)]
    report = research.compare_trade(trade(), data)
    assert report["outcomes"]["target_0.8r"] == {
        "status": "UNCHANGED_ACTUAL",
        "net": 200,
    }
    assert report["outcomes"]["lock_50pct"]["status"] == "UNRESOLVED"


def test_exchange_stop_gap_does_not_assume_guaranteed_fill() -> None:
    data = [tick(1, 101), tick(2, 111), tick(3, 110),
            tick(4, 85), tick(5, 85), tick(6, 85)]
    report = research.compare_trade(trade(), data)
    assert report["outcomes"]["lock_60pct"] == {
        "status": "UNRESOLVED",
        "reason": "stop_limit_gap_risk",
    }


def test_replay_archive_report_is_isolated_and_quality_gated(tmp_path) -> None:
    import json
    from datetime import datetime

    archive = ReplayArchive(tmp_path)
    archive.record(
        "snapshot", {"instruments": [{"instrument_token": 3}]},
        datetime.fromisoformat(ENTRY),
    )
    for second, bid in enumerate((101, 107, 109, 111, 108, 106), 1):
        event = tick(second, bid)
        archive.record("tick", event["payload"],
                       datetime.fromisoformat(event["available_at"]))
    archive.close()
    path = tmp_path / "2026-10-21.jsonl"
    trade_path = tmp_path / "trades.json"
    trade_path.write_text(json.dumps([trade()]), encoding="utf-8")
    report = research.run_exit_comparison(path, trade_path)
    assert report["trade_count"] == 1
    assert report["promotion_eligible"] is False
    assert report["live_settings_modified"] is False
    assert report["scenario_summary"]["target_0.6r"]["resolved"] == 1


def test_invalid_slippage_or_trade_geometry_fails_closed() -> None:
    with pytest.raises(ValueError, match="exit_replay_invalid_slippage"):
        research.compare_trade(trade(), [], slippage_points=-1)
    damaged = trade()
    damaged["initial_stop"] = None
    with pytest.raises(ValueError, match="exit_replay_invalid_trade_geometry"):
        research.compare_trade(damaged, [])


def test_replay_session_requires_ledger_trade_list(tmp_path) -> None:
    path = tmp_path / "rows.json"
    path.write_text('{"trade_id": "invalid"}', encoding="utf-8")
    with pytest.raises(ValueError, match="exit_replay_trade_list_required"):
        research.run_exit_comparison(tmp_path / "absent.jsonl", path)

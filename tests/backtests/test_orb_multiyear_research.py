"""Historical basket selection and settings selection must remain causal."""

import io
import json
import zipfile
from datetime import datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest
from scripts.research_orb_multiyear import (
    option_members,
    read_frame,
    select_atm_pair,
    select_candidate,
)


def option_frame(side, times):
    return pd.DataFrame(
        {
            "symbol": [f"{side} 10000"] * len(times),
            "date": ["2017-01-02"] * len(times),
            "time": times,
            "volume": [75] * len(times),
        }
    )


def test_opening_selection_cannot_use_later_contract_liquidity():
    available = {
        (10000, "CE"): option_frame("CE", ["09:16", "10:00"]),
        (10000, "PE"): option_frame("PE", ["10:00"]),
        (10050, "PE"): option_frame("PE", ["09:16"]),
    }
    assert select_atm_pair(10001, "2017-01-25", available, "2017-01-02") is None
    available[(10000, "PE")] = option_frame("PE", ["09:16", "10:00"])
    selected = select_atm_pair(10001, "2017-01-25", available, "2017-01-02")
    assert selected[0] == 10000
    assert select_atm_pair(10001, "2016-12-29", available, "2017-01-02") is None


def test_selection_rejects_tiny_stress_samples_and_missing_scenarios():
    results = [
        {
            "candidate": "stable",
            "slippage_bps_per_side": slip,
            "metrics": {"trade_count": 100, "expectancy": 4 - slip / 10},
            "data_quality": {"unresolved_exit_count": 0},
        }
        for slip in (10, 25, 50)
    ]
    results.extend(
        {
            "candidate": "tiny_peak",
            "slippage_bps_per_side": slip,
            "metrics": {"trade_count": 3, "expectancy": 10000},
            "data_quality": {"unresolved_exit_count": 0},
        }
        for slip in (10, 25, 50)
    )
    assert select_candidate(results) == "stable"
    assert select_candidate(results[:2]) is None
    contaminated = [dict(row) for row in results[:3]]
    contaminated[0] = {
        **contaminated[0],
        "data_quality": {"unresolved_exit_count": 1},
    }
    assert select_candidate(contaminated) is None


def test_nested_csv_and_txt_copies_are_not_double_counted():
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as archive:
        archive.writestr("CE 10000.txt", "a")
    outer = io.BytesIO()
    with zipfile.ZipFile(outer, "w") as archive:
        archive.writestr("TXT.zip", inner.getvalue())
        archive.writestr("CSV.zip", b"invalid duplicate representation")
    assert option_members(outer.getvalue()) == [("CE 10000.txt", b"a")]


def test_bad_option_rows_are_missing_not_repaired_and_spot_zero_volume_is_valid():
    raw = (
        b"CE 10000,2017/01/02,09:16,10,11,9,10,75\n"
        b"CE 10000,2017/01/02,09:17,10,11,9,20,75\n"
        b"CE 10000,2017/01/02,15:30,10,11,9,20,0\n"
    )
    frame = read_frame(io.BytesIO(raw), option=True)
    assert len(frame) == 1
    assert frame.attrs["invalid_rows_removed"] == 1
    assert frame.attrs["zero_volume_rows_removed"] == 1
    spot = read_frame(io.BytesIO(b"NIFTY,2019/01/02,09:16,10,11,9,10,0,0\n"))
    assert len(spot) == 1


def test_isolated_orb_replay_restores_environment_on_failure(monkeypatch, tmp_path):
    import nifty_scalper_bot.backtesting.strategy_research as research

    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("ORB_TARGET_RR", "1.8")
    monkeypatch.setattr(research, "load_archive", lambda directory: ({}, {}, "digest"))
    monkeypatch.setattr(research, "get_settings", lambda: SimpleNamespace(elite=None))

    def fail(*args, **kwargs):
        raise ValueError("failed replay")

    monkeypatch.setattr(research, "_scenario", fail)
    with pytest.raises(ValueError, match="failed replay"):
        research.run_orb_session_research(
            tmp_path, overrides={"ORB_TARGET_RR": "2.5"}, slippage_bps=10
        )
    import os

    assert os.environ["ORB_TARGET_RR"] == "1.8"
    with pytest.raises(ValueError, match="not_orb"):
        research.run_orb_session_research(
            tmp_path, overrides={"RISK_MAX_DAILY_LOSS": "0"}, slippage_bps=10
        )


def test_compact_context_preserves_actual_orb_trades(monkeypatch, tmp_path):
    from nifty_scalper_bot.backtesting.strategy_research import (
        _scenario,
        load_archive,
    )
    from nifty_scalper_bot.config.settings import get_settings

    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("STRATEGY_MODE", "directional_scalp")
    monkeypatch.setenv("ORB_ENABLED", "true")
    start = datetime.fromisoformat("2017-01-02T09:15:00+05:30")
    directory = tmp_path / "candles"
    directory.mkdir()
    for kind, symbol in (
        ("EQ", "NSE:NIFTY 50"),
        ("FUT", "NFO:NIFTY_F1_CONTEXT"),
        ("CE", "NFO:NIFTY2017012510000CE"),
        ("PE", "NFO:NIFTY2017012510000PE"),
    ):
        rows = []
        for minute in range(45):
            if kind in {"EQ", "FUT"}:
                price = 100 if minute < 15 else 104 + (minute - 15) * 0.1
                opening = 100 if minute == 15 else price
                high, low = max(price, opening) + 0.2, min(price, opening) - 0.2
                volume = 5000 if minute == 15 else 1000
            else:
                price = 100 + max(0, minute - 15) * 0.3
                opening, high, low, volume = price, price + 0.1, price - 0.1, 750
            rows.append(
                [
                    (start + timedelta(minutes=minute)).isoformat(),
                    opening,
                    high,
                    low,
                    price,
                    volume,
                ]
            )
        (directory / f"{kind}.json").write_text(
            json.dumps(
                {
                    "symbol": symbol,
                    "timestamp_convention": "bar_start",
                    "instrument": {
                        "name": "NIFTY",
                        "instrument_type": kind,
                        "expiry": "2017-01-25",
                        "lot_size": 75,
                    },
                    "candles": rows,
                }
            )
        )
    histories, instruments, _ = load_archive(tmp_path)
    baseline = _scenario(
        histories, instruments, get_settings().elite, 0, components={"ORBPro"}
    )["strategies"]["ORBPro"]
    compact = _scenario(
        histories,
        instruments,
        get_settings().elite,
        0,
        components={"ORBPro"},
        compact_orb_context=True,
    )["strategies"]["ORBPro"]
    assert baseline["metrics"]["trade_count"] > 0
    assert compact["metrics"] == baseline["metrics"]

    economic_fields = (
        "entry_time",
        "entry_price",
        "stop_loss",
        "take_profit",
        "quantity",
        "symbol",
        "exit_time",
        "exit_price",
        "exit_reason",
        "gross_pnl",
        "fees",
        "net_pnl",
    )
    assert [
        {field: trade[field] for field in economic_fields}
        for trade in compact["trades"]
    ] == [
        {field: trade[field] for field in economic_fields}
        for trade in baseline["trades"]
    ]
    assert (
        compact["trades"][0]["raw_setup_score"]
        >= baseline["trades"][0]["raw_setup_score"]
    )
    assert "underlying_direction_alignment" in compact["trades"][0]["score_reasons"]
    assert "futures_vwap_slope_alignment" in compact["trades"][0]["score_reasons"]

    first = baseline["trades"][0]
    entry_time = datetime.fromisoformat(first["entry_time"])
    history = histories[first["symbol"]]
    missing_time = entry_time + timedelta(minutes=1)
    removed = history.pop(missing_time)
    strict = _scenario(
        histories,
        instruments,
        get_settings().elite,
        0,
        components={"ORBPro"},
        compact_orb_context=True,
        strict_liquidity=True,
    )["strategies"]["ORBPro"]
    assert strict["metrics"]["trade_count"] == 0
    assert strict["data_quality"]["unresolved_exit_count"] == 1
    unresolved_reason = strict["unresolved_trades"][0]["unresolved_reason"]
    assert unresolved_reason == "history_gap_unresolved"
    stress_reasons = strict["worst_case_stress_exit_reasons"]
    assert stress_reasons["unpriced_gap_full_premium_stress"] == 1
    assert strict["worst_case_stress_trades"][0]["exit_price"] == 0.01
    history[missing_time] = removed
    history[entry_time]["volume"] = 0
    strict = _scenario(
        histories,
        instruments,
        get_settings().elite,
        0,
        components={"ORBPro"},
        compact_orb_context=True,
        strict_liquidity=True,
    )["strategies"]["ORBPro"]
    assert strict["no_vote_reasons"]["next_minute_has_no_trades"] == 1
    assert strict["metrics"]["trade_count"] == 0


def test_lifecycle_proxy_ratchets_stop_from_completed_bar_only():
    from nifty_scalper_bot.backtesting.strategy_research import (
        _apply_bar_lifecycle_proxy,
    )

    opened = datetime(2026, 1, 5, 10, 0)
    position = {
        "entry_time": opened,
        "entry_price": 100.0,
        "stop_loss": 95.0,
        "initial_stop_loss": 95.0,
        "take_profit": 112.5,
        "quantity": 75,
        "high_water": 100.0,
        "trail_updates": 0,
    }
    bar = {
        "timestamp": opened + timedelta(minutes=5),
        "open": 104.0,
        "high": 110.0,
        "low": 103.0,
        "close": 109.0,
        "volume": 100.0,
    }
    changed = _apply_bar_lifecycle_proxy(position, bar, prior_atr=2.0)
    assert changed is True
    assert position["stop_loss"] > 100.0
    assert position["stop_loss"] < bar["high"]
    assert position["trail_updates"] == 1


def test_lifecycle_time_stop_matches_12_minute_half_r_progress_rule():
    from nifty_scalper_bot.backtesting.strategy_research import (
        _bar_lifecycle_time_stop_due,
    )

    opened = datetime(2026, 1, 5, 10, 0)
    position = {
        "entry_time": opened,
        "entry_price": 100.0,
        "stop_loss": 95.0,
        "initial_stop_loss": 95.0,
        "take_profit": 112.5,
        "quantity": 75,
        "high_water": 102.0,
        "trail_updates": 0,
    }
    assert (
        _bar_lifecycle_time_stop_due(position, opened + timedelta(minutes=11)) is False
    )
    assert (
        _bar_lifecycle_time_stop_due(position, opened + timedelta(minutes=12)) is True
    )
    position["high_water"] = 102.5
    assert (
        _bar_lifecycle_time_stop_due(position, opened + timedelta(minutes=12)) is False
    )

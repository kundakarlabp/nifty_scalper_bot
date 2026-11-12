"""Causal Breeze research replay tests; no broker, no real orders.

Ownership: Offline research-only, with synthetic source schemas and candles.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from scripts.replay_breeze_archive_sqlite import execute, prepare_causal_day


def _databases(root: Path, *, omit_ce: bool = False) -> tuple[Path, Path]:
    minute = root / "intraday.sqlite"
    eod = root / "eod.sqlite"
    with sqlite3.connect(eod) as conn:
        conn.executescript("""
            CREATE TABLE sessions (day TEXT, available_not_before TEXT);
            CREATE TABLE daily_index (
                day TEXT, available_not_before TEXT, close REAL
            );
            CREATE TABLE contract_eod (
                day TEXT, kind TEXT, expiry TEXT, strike REAL,
                lot_size INTEGER, instrument_name TEXT
            );
            """)
        conn.execute("INSERT INTO sessions VALUES (?,?)", ("2025-11-03", "2025-11-04"))
        conn.execute(
            "INSERT INTO daily_index VALUES (?,?,?)",
            ("2025-11-03", "2025-11-04", 22530.0),
        )
        conn.execute(
            "INSERT INTO contract_eod VALUES (?,?,?,?,?,?)",
            ("2025-11-03", "FUT", "2025-11-25", None, 75, "NIFTY25NOVFUT"),
        )
        for strike in (22500, 22600):
            for kind in ("CE", "PE"):
                conn.execute(
                    "INSERT INTO contract_eod VALUES (?,?,?,?,?,?)",
                    (
                        "2025-11-03",
                        kind,
                        "2025-11-11",
                        strike,
                        75,
                        f"NIFTY25NOV{strike}{kind}",
                    ),
                )
        # Same-day EOD is deliberately unavailable at replay time and must
        # not bias selected contract or previous close.
        conn.execute(
            "INSERT INTO sessions VALUES (?,?)",
            ("2025-11-04", "2025-11-05"),
        )
        conn.execute(
            "INSERT INTO daily_index VALUES (?,?,?)",
            ("2025-11-04", "2025-11-05", 30000.0),
        )
        conn.commit()
    with sqlite3.connect(minute) as conn:
        conn.executescript("""
            CREATE TABLE series (
                id INTEGER PRIMARY KEY, trade_day TEXT, contract_key TEXT,
                status TEXT, observed_minutes INTEGER, missing_minutes INTEGER,
                quarantined_source_minutes INTEGER
            );
            CREATE TABLE candles (
                series_id INTEGER, stamp TEXT, open REAL, high REAL,
                low REAL, close REAL, volume INTEGER, oi INTEGER
            );
            """)
        contracts = [
            "NIFTY_INDEX",
            "NIFTY_2025-11-25_FUT",
            "NIFTY_2025-11-11_22500_CE",
            "NIFTY_2025-11-11_22500_PE",
            "NIFTY_2025-11-11_22600_CE",
            "NIFTY_2025-11-11_22600_PE",
        ]
        for n, key in enumerate(contracts, start=1):
            if omit_ce and key.endswith("22500_CE"):
                continue
            conn.execute(
                "INSERT INTO series VALUES (?,?,?,?,?,?,?)",
                (n, "2025-11-04", key, "PARTIAL", 4, 371, 0),
            )
            for minute_index in range(4):
                stamp = datetime(
                    2025,
                    11,
                    4,
                    9,
                    15,
                    tzinfo=timezone(timedelta(hours=5, minutes=30)),
                ) + timedelta(minutes=minute_index)
                value = 100.0 + minute_index
                conn.execute(
                    "INSERT INTO candles VALUES (?,?,?,?,?,?,?,?)",
                    (
                        n,
                        stamp.isoformat(),
                        value,
                        value + 1,
                        value - 1,
                        value + 0.5,
                        100,
                        1000,
                    ),
                )
        conn.commit()
    return minute, eod


def test_prepare_only_uses_prior_day_not_same_day(tmp_path: Path) -> None:
    minute, eod = _databases(tmp_path)
    out = tmp_path / "shadow"
    result = execute(minute, eod, "2025-11-04", out, prepare_only=True)
    assert result["status"] == "RESEARCH_BAR_PROXY_READY"
    assert result["prior_eod_day"] == "2025-11-03"
    assert result["prior_eod_close_used_for_selection"] == 22530.0
    assert result["selected_contracts"]["CE"]["contract_key"].endswith("22500_CE")
    assert result["execution_grade_ready"] is False
    assert not result["historical_bid_ask_available"]
    from nifty_scalper_bot.backtesting.strategy_research import load_archive

    histories, instruments, _ = load_archive(out)
    assert len(histories) == 4
    assert sorted(
        row["lot_size"]
        for row in instruments.values()
        if row["instrument_type"] in ("CE", "PE")
    ) == [75, 75]
    with (out / "candles" / "CE.json").open() as source:
        ce = json.load(source)
    assert ce["historical_broker_token_verified"] is False
    assert len(ce["candles"]) == 4
    assert (out / "selection_report.json").exists()


def test_absent_true_atm_pair_cannot_substitute_later_available_strike(
    tmp_path: Path,
) -> None:
    minute, eod = _databases(tmp_path, omit_ce=True)
    out = tmp_path / "shadow"
    result = prepare_causal_day(minute, eod, "2025-11-04", out)
    assert result["status"] == "ABSTAIN"
    assert result["reason"] == "selected_ce_intraday_unavailable"
    assert not out.exists()


def test_no_previous_master_abstains_before_output(tmp_path: Path) -> None:
    minute, eod = _databases(tmp_path)
    out = tmp_path / "shadow"
    result = prepare_causal_day(minute, eod, "2025-11-03", out)
    assert result["reason"] == "prior_eod_unavailable"
    assert not out.exists()


def test_reject_live_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    minute, eod = _databases(tmp_path)
    monkeypatch.setenv("ENABLE_LIVE", "true")
    with pytest.raises(ValueError, match="breeze_replay_refuses_live_environment"):
        execute(
            minute,
            eod,
            "2025-11-04",
            tmp_path / "nope",
            prepare_only=True,
        )
    assert not (tmp_path / "nope").exists()


def test_reject_existing_output(tmp_path: Path) -> None:
    minute, eod = _databases(tmp_path)
    out = tmp_path / "existing"
    out.mkdir()
    with pytest.raises(ValueError, match="research_day_output_exists"):
        prepare_causal_day(minute, eod, "2025-11-04", out)


def test_real_shadow_component_engine_consumes_selected_breeze_bars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the existing research component engine, not a mocked callback."""
    minute, eod = _databases(tmp_path)
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("ENABLE_LIVE", "false")
    for name in ("SMC_ENABLED", "VWAP_PRO_ENABLED", "ORB_ENABLED"):
        monkeypatch.setenv(name, "true")
    result = execute(minute, eod, "2025-11-04", tmp_path / "shadow")
    assert result["status"] == "RESEARCH_BAR_PROXY_REPLAYED"
    output = tmp_path / "shadow" / "research_scenarios.json"
    report = json.loads(output.read_text())
    assert report["breeze_evidence_label"] == "RESEARCH_CANDIDATE_BAR_PROXY_ONLY"
    assert report["execution_grade_ready"] is False
    assert report["promotion_eligible"] is False
    assert len(report["scenarios"]) >= 3

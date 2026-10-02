"""Archived research must be causal and explicit about simulated execution."""

import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def isolated_research_mode(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("STRATEGY_MODE", "directional_scalp")
    for name in ("SMC_ENABLED", "VWAP_PRO_ENABLED", "ORB_ENABLED"):
        monkeypatch.setenv(name, "true")


def archive(tmp_path, *, gap=False):
    candles = tmp_path / "candles"
    candles.mkdir()
    first = datetime.fromisoformat("2026-10-01T09:30:00+05:30")
    for symbol, kind in (
        ("NSE:NIFTY 50", "EQ"),
        ("NFO:NIFTY26OCTFUT", "FUT"),
        ("NFO:NIFTY26O0622400CE", "CE"),
        ("NFO:NIFTY26O0622400PE", "PE"),
    ):
        rows = []
        for minute in range(4):
            if gap and minute == 1 and kind == "CE":
                continue
            rows.append(
                [
                    (first + timedelta(minutes=minute)).isoformat(),
                    100,
                    106,
                    94,
                    100,
                    1000,
                ]
            )
        payload = {
            "symbol": symbol,
            "timestamp_convention": "bar_start",
            "instrument": {
                "instrument_type": kind,
                "name": "NIFTY",
                "lot_size": 75,
                "expiry": "2026-10-06",
            },
            "candles": rows,
        }
        (candles / f"{kind}.json").write_text(json.dumps(payload))
    return first


def test_component_replay_uses_completed_history_next_open_and_stop_first(
    tmp_path, monkeypatch
):
    from nifty_scalper_bot.backtesting.strategy_research import run_archived_research

    first = archive(tmp_path)
    observed = []

    class Strategy:
        name = "ORBPro"
        config = SimpleNamespace(enabled=True)

        def generate_signal(self, symbol, indicators, current_price, position=None):
            observed.append(indicators["history_count"])
            if indicators["history_count"] == 1:
                return SimpleNamespace(
                    action="BUY", stop_loss=95, take_profit=105, metadata={}
                )
            return None

        @property
        def evaluation_health(self):
            return {"healthy": True}

        def notify_entry_accepted(self, side, *, setup_id=None):
            pass

    monkeypatch.setattr(
        "nifty_scalper_bot.backtesting.strategy_research.build_elite_strategies",
        lambda settings, engine: [Strategy()],
    )
    report = run_archived_research(tmp_path, slippage_bps=0)
    assert report["scope"] == "active_contract_strategy_components"
    assert report["live_equivalent"] is False
    trades = report["scenarios"][0]["strategies"]["ORBPro"]["trades"]
    assert len(trades) == 2
    assert trades[0]["entry_time"] == (first + timedelta(minutes=1)).isoformat()
    assert trades[0]["exit_reason"] == "stop"
    assert trades[0]["exit_price"] == 95
    assert trades[0]["fees"] > 0
    assert trades[0]["net_pnl"] < trades[0]["gross_pnl"]
    assert observed[:2] == [1, 1]


def test_missing_next_minute_never_creates_a_delayed_entry(tmp_path, monkeypatch):
    from nifty_scalper_bot.backtesting.strategy_research import run_archived_research

    archive(tmp_path, gap=True)

    class Strategy:
        name = "SMC"
        config = SimpleNamespace(enabled=True)

        def generate_signal(self, symbol, indicators, current_price, position=None):
            if indicators["history_count"] == 1:
                return SimpleNamespace(
                    action="BUY", stop_loss=95, take_profit=105, metadata={}
                )
            return None

        @property
        def evaluation_health(self):
            return {"healthy": True}

        def notify_entry_accepted(self, side, *, setup_id=None):
            pass

    monkeypatch.setattr(
        "nifty_scalper_bot.backtesting.strategy_research.build_elite_strategies",
        lambda settings, engine: [Strategy()],
    )
    report = run_archived_research(tmp_path, slippage_bps=0)
    trades = report["scenarios"][0]["strategies"]["SMC"]["trades"]
    assert [trade["symbol"] for trade in trades] == ["NFO:NIFTY26O0622400PE"]


def test_invalid_archive_fails_instead_of_reporting_empty_success(tmp_path):
    from nifty_scalper_bot.backtesting.strategy_research import run_archived_research

    with pytest.raises(ValueError, match="research_history_unavailable"):
        run_archived_research(tmp_path)


def test_real_production_components_run_deterministically_and_reject_live_process(
    tmp_path, monkeypatch
):
    from nifty_scalper_bot.backtesting.strategy_research import run_archived_research

    archive(tmp_path)
    # Extend into actual strategy warm-up and opening-range coverage.
    for path in (tmp_path / "candles").glob("*.json"):
        payload = json.loads(path.read_text())
        first = datetime.fromisoformat("2026-10-01T09:15:00+05:30")
        payload["candles"] = [
            [
                (first + timedelta(minutes=minute)).isoformat(),
                100 + minute / 10,
                102 + minute / 10,
                99 + minute / 10,
                101 + minute / 10,
                1000 + minute * 10,
            ]
            for minute in range(65)
        ]
        path.write_text(json.dumps(payload))
    report = run_archived_research(tmp_path)
    assert report == run_archived_research(tmp_path)
    strategies = report["scenarios"][0]["strategies"]
    assert set(strategies) == {"ORBPro", "SMC", "VWAPPro"}
    assert all(result["evaluations"] > 0 for result in strategies.values())
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    with pytest.raises(ValueError, match="research_requires_shadow_process"):
        run_archived_research(tmp_path)


@pytest.mark.parametrize("defect", ["conflicting_duplicate", "expired", "not_nifty"])
def test_archive_identity_and_duplicate_defects_fail_closed(tmp_path, defect):
    from nifty_scalper_bot.backtesting.strategy_research import run_archived_research

    archive(tmp_path)
    path = tmp_path / "candles/CE.json"
    payload = json.loads(path.read_text())
    if defect == "conflicting_duplicate":
        duplicate = payload["candles"][0].copy()
        duplicate[4] = 101
        payload["candles"].append(duplicate)
    elif defect == "expired":
        payload["instrument"]["expiry"] = "2026-09-30"
    else:
        payload["instrument"]["name"] = "BANKNIFTY"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        run_archived_research(tmp_path)

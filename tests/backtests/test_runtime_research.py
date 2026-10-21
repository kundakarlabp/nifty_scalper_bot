import pytest

from nifty_scalper_bot.backtesting.runtime_research import validate_snapshot


def test_full_runtime_rejects_missing_identity_and_live_positions():
    with pytest.raises(ValueError, match="replay_instrument_catalog_missing"):
        validate_snapshot({"basket": {}, "initial_balance": 100000})
    with pytest.raises(ValueError, match="replay_initial_positions_unsupported"):
        validate_snapshot(
            {
                "instruments": [{"instrument_token": 1}],
                "initial_positions": [{"quantity": 1}],
            }
        )


def test_runtime_requires_captured_effective_configuration():
    with pytest.raises(ValueError, match="replay_configuration_missing"):
        validate_snapshot(
            {
                "instruments": [{"instrument_token": 1}],
                "initial_positions": [],
                "initial_balance": 100000,
                "basket": {"selected_ce": "CE"},
            }
        )


def test_replay_epoch_preflight_rejects_multiple_process_runs(tmp_path):
    from datetime import datetime, timedelta

    from nifty_scalper_bot.backtesting.runtime_research import validate_replay_epochs
    from nifty_scalper_bot.storage.replay_archive import ReplayArchive

    now = datetime.fromisoformat("2026-10-01T10:00:00+05:30")
    snapshot = {
        "instruments": [{"instrument_token": 100}],
        "basket": {"selected_ce": "CE"},
        "effective_settings": {"risk": {"enabled": True}},
        "runner_config": {"some_value": True},
        "initial_balance": 100000.0,
        "initial_positions": [],
    }
    first = ReplayArchive(tmp_path)
    first.record("snapshot", snapshot, now)
    first.close()
    second = ReplayArchive(tmp_path)
    second.record("snapshot", snapshot, now + timedelta(minutes=1))
    second.close()

    with pytest.raises(ValueError, match="replay_multiple_runs_unrestored_state"):
        validate_replay_epochs(tmp_path / "2026-10-01.jsonl")


def test_replay_epoch_preflight_rejects_changed_settings_but_allows_basket(
    tmp_path,
):
    from datetime import datetime, timedelta

    from nifty_scalper_bot.backtesting.runtime_research import validate_replay_epochs
    from nifty_scalper_bot.storage.replay_archive import ReplayArchive

    now = datetime.fromisoformat("2026-10-01T10:00:00+05:30")
    snapshot = {
        "instruments": [{"instrument_token": 100}],
        "basket": {"selected_ce": "CE"},
        "effective_settings": {"risk": {"enabled": True}},
        "runner_config": {"some_value": True},
        "initial_balance": 100000.0,
        "initial_positions": [],
    }
    archive = ReplayArchive(tmp_path)
    archive.record("snapshot", snapshot, now)
    archive.record(
        "snapshot",
        {**snapshot, "basket": {"selected_ce": "CE", "selected_pe": "PE"}},
        now + timedelta(minutes=1),
    )
    archive.close()
    path = tmp_path / "2026-10-01.jsonl"
    count, first_snapshot, snapshots = validate_replay_epochs(path)
    assert (count, snapshots) == (2, 2)
    assert first_snapshot == snapshot

    other = ReplayArchive(tmp_path / "changed")
    other.record("snapshot", snapshot, now)
    other.record(
        "snapshot",
        {**snapshot, "effective_settings": {"risk": {"enabled": False}}},
        now + timedelta(minutes=1),
    )
    other.close()
    with pytest.raises(ValueError, match="replay_snapshot_configuration_changed"):
        validate_replay_epochs(tmp_path / "changed/2026-10-01.jsonl")


def test_recorded_feed_bootstraps_real_production_owners(tmp_path, monkeypatch):
    import json
    from datetime import datetime, timedelta

    import nifty_scalper_bot.core.app as app
    from nifty_scalper_bot.backtesting.runtime_research import run_runtime_session
    from nifty_scalper_bot.config.settings import get_settings
    from nifty_scalper_bot.storage.replay_archive import ReplayArchive, capture_settings
    from nifty_scalper_bot.utils.serialization import to_json_safe

    monkeypatch.setenv("REPLAY_ISOLATED_PROCESS", "true")
    settings = get_settings()
    now = datetime.fromisoformat("2026-10-01T10:00:00+05:30")
    ce = "NFO:NIFTY26O0125000CE"
    pe = "NFO:NIFTY26O0125000PE"
    future = "NFO:NIFTY26OCTFUT"
    symbols = ["NSE:NIFTY 50", future, ce, pe]
    tokens = [256265, 101, 102, 103]
    basket = {
        "option_symbols": [ce, pe],
        "selected_ce": ce,
        "selected_pe": pe,
        "selected_ce_token": 102,
        "selected_pe_token": 103,
        "futures_symbol": future,
        "futures_token": 101,
        "option_expiry": "2026-10-01",
        "expiry": "2026-10-01",
        "all_tokens": tokens,
        "all_symbols": symbols,
        "token_by_symbol": dict(zip(symbols, tokens)),
        "basket_version": "test-v1",
    }
    instruments = [
        {
            "instrument_token": token,
            "tradingsymbol": symbol.split(":")[-1],
            "exchange": symbol.split(":")[0],
            "name": "NIFTY",
            "instrument_type": kind,
            "expiry": "2026-10-01",
            "strike": 25000 if kind in {"CE", "PE"} else 0,
            "lot_size": 65,
            "tick_size": 0.05,
        }
        for symbol, token, kind in zip(symbols, tokens, ["EQ", "FUT", "CE", "PE"])
    ]
    archive = ReplayArchive(tmp_path / "archive")
    archive.record(
        "snapshot",
        {
            "instruments": instruments,
            "basket": basket,
            "effective_settings": capture_settings(settings),
            "runner_config": to_json_safe(app._get_strategy_config(settings.app)),
            "initial_balance": 100000,
            "initial_positions": [],
            "history": {},
        },
        now,
    )
    for symbol, token in zip(symbols, tokens):
        archive.record(
            "tick",
            {
                "symbol": symbol,
                "instrument_token": token,
                "timestamp": now.isoformat(),
                "exchange_timestamp": now.isoformat(),
                "received_at": now.timestamp(),
                "ltp": 100,
                "last_price": 100,
                "bid": 99.95,
                "ask": 100.05,
                "volume": 1000,
                "source": "ws",
                "depth": {
                    "buy": [{"price": 99.95, "quantity": 650}],
                    "sell": [{"price": 100.05, "quantity": 650}],
                },
            },
            now,
        )
    for symbol, token in zip(symbols, tokens):
        later = now + timedelta(minutes=1)
        archive.record(
            "tick",
            {
                "symbol": symbol,
                "instrument_token": token,
                "timestamp": later.isoformat(),
                "exchange_timestamp": later.isoformat(),
                "received_at": later.timestamp(),
                "ltp": 101,
                "last_price": 101,
                "bid": 100.95,
                "ask": 101.05,
                "volume": 1000,
                "volume_delta": 1000,
                "source": "ws",
                "depth": {
                    "buy": [{"price": 100.95, "quantity": 650}],
                    "sell": [{"price": 101.05, "quantity": 650}],
                },
            },
            later,
        )
    archive.close()
    report = run_runtime_session(
        tmp_path / "archive/2026-10-01.jsonl", tmp_path / "run"
    )
    assert report["events_processed"] == 9
    assert all(count >= 1 for count in report["candle_counts"].values())
    assert report["runtime_owners"]["strategy_runner"] == "StrategyRunner"
    assert report["live_equivalent"] is False
    assert report["orders"] == []  # insufficient warmup must not be bypassed
    assert json.loads((tmp_path / "run/report.json").read_text())["scope"].startswith(
        "production_composition"
    )


# Quote-level exit research regressions reuse this canonical replay test module.


def _exit_research():
    from nifty_scalper_bot.backtesting import exit_policy_research

    return exit_policy_research


def _exit_trade():
    return {
        "trade_id": "test",
        "symbol": "NFO:NIFTY26OCT23100CE",
        "side": "BUY",
        "quantity": 65,
        "entry_price": 100,
        "initial_stop": 90,
        "net_pnl": 200,
        "effective_costs": 70,
        "ledger_complete": True,
        "entry_filled_at": "2026-10-21T04:03:00+00:00",
        "exit_filled_at": "2026-10-21T04:03:06+00:00",
    }


def _exit_quote(second: int, bid: float, *, qty: int = 65):
    return {
        "kind": "tick",
        "available_at": f"2026-10-21T04:03:{second:02d}+00:00",
        "payload": {
            "symbol": "NFO:NIFTY26OCT23100CE",
            "ltp": bid + 1,
            "depth": {"buy": [{"price": bid, "quantity": qty}]},
        },
    }


def test_exit_research_targets_require_executable_bid() -> None:
    quotes = [
        _exit_quote(1, 101),
        _exit_quote(2, 106),
        _exit_quote(3, 108.2),
        _exit_quote(4, 110.2),
        _exit_quote(5, 109),
        _exit_quote(6, 107),
    ]
    result = _exit_research().compare_trade(_exit_trade(), quotes)
    assert result["coverage_ready"] is True
    assert result["outcomes"]["target_0.6r"]["net"] == 287.5
    assert result["outcomes"]["target_0.8r"]["net"] == 417.5
    assert result["outcomes"]["target_1.0r"]["net"] == 547.5


def test_exit_research_tier2_locks_ratchet_only_after_peak() -> None:
    bids = [101, 105, 111, 108, 105, 104]
    quotes = [_exit_quote(i, price) for i, price in enumerate(bids, 1)]
    result = _exit_research().compare_trade(_exit_trade(), quotes)
    assert result["outcomes"]["lock_40pct"]["status"] == "MODELED_STOP"
    assert result["outcomes"]["lock_50pct"]["status"] == "MODELED_STOP"
    assert result["outcomes"]["lock_60pct"]["at"].endswith("04:03:05+00:00")


def test_exit_research_rejects_ltp_without_sufficient_depth() -> None:
    quotes = [_exit_quote(i, 120, qty=1) for i in range(1, 7)]
    result = _exit_research().compare_trade(_exit_trade(), quotes)
    assert result["coverage_ready"] is False
    assert all(
        value["status"] == "UNRESOLVED"
        for value in result["outcomes"].values()
    )


def test_exit_research_future_quotes_cannot_trigger_past_targets() -> None:
    quotes = [
        _exit_quote(i, price)
        for i, price in enumerate((100, 101, 102, 103, 102, 101, 120), 1)
    ]
    result = _exit_research().compare_trade(_exit_trade(), quotes)
    assert result["outcomes"]["target_0.8r"]["status"] == "UNCHANGED_ACTUAL"
    assert result["outcomes"]["lock_50pct"]["status"] == "UNRESOLVED"


def test_exit_research_stop_limit_gap_cannot_claim_fill() -> None:
    quotes = [_exit_quote(i, bid) for i, bid in enumerate(
        (101, 111, 110, 85, 85, 85), 1
    )]
    result = _exit_research().compare_trade(_exit_trade(), quotes)
    assert result["outcomes"]["lock_60pct"]["reason"] == "stop_limit_gap_risk"


def test_exit_research_invalid_trade_geometry_or_slippage_fails_closed() -> None:
    trade = _exit_trade()
    trade["initial_stop"] = None
    with pytest.raises(ValueError, match="exit_replay_invalid_trade_geometry"):
        _exit_research().compare_trade(trade, [])
    with pytest.raises(ValueError, match="exit_replay_invalid_slippage"):
        _exit_research().compare_trade(_exit_trade(), [], slippage_points=-1)

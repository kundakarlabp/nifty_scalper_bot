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

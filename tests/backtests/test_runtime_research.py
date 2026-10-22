import json
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.backtesting.runtime_research import validate_snapshot
from nifty_scalper_bot.storage import replay_archive as archive

worker = import_module("scripts.run_runtime_replay")


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
    assert all(value["status"] == "UNRESOLVED" for value in result["outcomes"].values())


def test_exit_research_future_quotes_cannot_trigger_past_targets() -> None:
    quotes = [
        _exit_quote(i, price)
        for i, price in enumerate((100, 101, 102, 103, 102, 101, 120), 1)
    ]
    result = _exit_research().compare_trade(_exit_trade(), quotes)
    assert result["outcomes"]["target_0.8r"]["status"] == "UNCHANGED_ACTUAL"
    assert result["outcomes"]["lock_50pct"]["status"] == "UNRESOLVED"


def test_exit_research_stop_limit_gap_cannot_claim_fill() -> None:
    quotes = [
        _exit_quote(i, bid) for i, bid in enumerate((101, 111, 110, 85, 85, 85), 1)
    ]
    result = _exit_research().compare_trade(_exit_trade(), quotes)
    assert result["outcomes"]["lock_60pct"]["reason"] == "stop_limit_gap_risk"


def test_exit_research_invalid_trade_geometry_or_slippage_fails_closed() -> None:
    trade = _exit_trade()
    trade["initial_stop"] = None
    with pytest.raises(ValueError, match="exit_replay_invalid_trade_geometry"):
        _exit_research().compare_trade(trade, [])
    with pytest.raises(ValueError, match="exit_replay_invalid_slippage"):
        _exit_research().compare_trade(_exit_trade(), [], slippage_points=-1)


def _paired_report(*, net=200.0, price=102.0, symbol="NFO:NIFTY26OCT23100CE"):
    return {
        "archive_sha256": "verified_sha",
        "events_processed": 12,
        "snapshots_processed": 1,
        "captured_revision": "run_sha",
        "positions": [],
        "orders": [
            {
                "transaction_type": "BUY",
                "symbol": symbol,
                "status": "complete",
                "filled_quantity": 65,
                "average_price": price,
                "timestamp": 101.0,
            }
        ],
        "modeled_account": {"net_equity_pnl": net, "fees": 74.0},
        "max_equity_drawdown": 45.0,
    }


def _paired_fidelity():
    return {
        "archive_integrity_verified": True,
        "observed_selected_context_available": True,
        "capture_sha256": "verified_sha",
    }


def test_paired_adaptive_runtime_comparison_uses_identical_completed_entries():
    observed = _paired_report(net=200)
    adaptive = _paired_report(net=260)
    comparison = worker.compare_paired_reports(
        observed,
        adaptive,
        _paired_fidelity(),
        expected_sha="verified_sha",
        slippage_bps=25,
    )
    assert comparison["state"] == "MODELED_PAIRED_COMPARISON"
    assert comparison["net_delta"] == 60
    assert comparison["entry_parity"] is True
    assert comparison["promotion_eligible"] is False


def test_paired_adaptive_runtime_blocks_lookahead_and_different_entries():
    observed = _paired_report()
    divergent = _paired_report(symbol="NFO:NIFTY26OCT23000CE")
    comparison = worker.compare_paired_reports(
        observed,
        divergent,
        _paired_fidelity(),
        expected_sha="verified_sha",
        slippage_bps=10,
    )
    assert comparison["state"] == "UNRESOLVED"
    assert comparison["net_delta"] is None
    assert comparison["entry_parity"] is False

    bad_sha = worker.compare_paired_reports(
        observed,
        observed,
        _paired_fidelity(),
        expected_sha="tampered",
        slippage_bps=10,
    )
    assert bad_sha["blocker"] == "paired_replay_input_parity_mismatch"


def test_paired_adaptive_runtime_blocks_missing_depth_and_open_positions():
    row = _paired_report()
    poor_fidelity = {**_paired_fidelity(), "observed_selected_context_available": False}
    invalid = worker.compare_paired_reports(
        row, row, poor_fidelity, expected_sha="verified_sha", slippage_bps=10
    )
    assert invalid["blocker"] == "incomplete_captured_quote_fidelity"
    has_position = {**row, "positions": [{"quantity": 65}]}
    exposed = worker.compare_paired_reports(
        row,
        has_position,
        _paired_fidelity(),
        expected_sha="verified_sha",
        slippage_bps=10,
    )
    assert exposed["blocker"] == "paired_replay_open_position"


def test_paired_adaptive_rejects_unfinished_and_empty_buy_fills():
    row = _paired_report()
    partial = _paired_report()
    partial["orders"][0]["status"] = "open"
    result = worker.compare_paired_reports(
        row,
        partial,
        _paired_fidelity(),
        expected_sha="verified_sha",
        slippage_bps=10,
    )
    assert result["blocker"] == "paired_replay_partial_entry"
    no_entries = _paired_report()
    no_entries["orders"] = []
    result = worker.compare_paired_reports(
        no_entries,
        no_entries,
        _paired_fidelity(),
        expected_sha="verified_sha",
        slippage_bps=10,
    )
    assert result["blocker"] == "paired_replay_entry_divergence_or_no_fills"


def test_paired_run_uses_sanitized_isolated_offline_workers(monkeypatch, tmp_path):
    session = tmp_path / "2026-10-21.jsonl"
    session.write_text("immutable archive")
    monkeypatch.setenv("REAL_BROKER_PASSWORD", "do-not-inherit")
    monkeypatch.setattr(archive, "archive_fingerprint", lambda _path: "verified_sha")
    monkeypatch.setattr(
        worker, "write_capture_preflight", lambda *_: _paired_fidelity()
    )
    calls = []

    def fake_run(command, *, cwd, env, stdout, stderr, timeout, check):
        del cwd, stdout, stderr, timeout, check
        calls.append((env.copy(), command[:]))
        assert env["ALLOW_NETWORK"] == "false"
        assert "REAL_BROKER_PASSWORD" not in env
        output = command[command.index("--output") + 1]
        report = _paired_report(
            net=260 if env["ADAPTIVE_PROFIT_LOCK_ENABLED"] == "true" else 200
        )
        Path(output, "report.json").write_text(json.dumps(report))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(worker.subprocess, "run", fake_run)
    report = worker.run_paired_replays(session, tmp_path / "out", stress_bps=(10, 25))
    assert report["state"] == "MODELED_COMPARISON_ONLY"
    assert [x[0]["ADAPTIVE_PROFIT_LOCK_ENABLED"] for x in calls] == [
        "false",
        "true",
        "false",
        "true",
    ]
    assert [r["net_delta"] for r in report["stress"]] == [60, 60]
    assert report["promotion_eligible"] is False


def test_paired_run_missing_capture_refuses_subprocess(monkeypatch, tmp_path):
    monkeypatch.setattr(
        worker,
        "write_capture_preflight",
        lambda *_: {"archive_integrity_verified": False},
    )
    monkeypatch.setattr(
        worker.subprocess,
        "run",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("must not run with invalid archive")
        ),
    )
    report = worker.run_paired_replays(tmp_path / "missing.jsonl", tmp_path / "output")
    assert report["state"] == "BLOCKED"
    assert report["blocker"] == "no_usable_archived_depth_session"


def test_verified_replay_process_epochs_are_independently_replayable(tmp_path):
    from datetime import datetime, timedelta

    from nifty_scalper_bot.backtesting.runtime_research import (
        materialize_replay_epochs,
        validate_replay_epochs,
    )
    from nifty_scalper_bot.storage.replay_archive import ReplayArchive, iter_session

    base = datetime.fromisoformat("2026-10-21T09:30:00+05:30")
    snapshot = {
        "instruments": [{"instrument_token": 123}],
        "effective_settings": {"risk": {"enabled": True}},
        "runner_config": {"mode": "test"},
        "basket": {"selected_ce": "NFO:NIFTY_CE"},
        "initial_balance": 100000,
        "initial_positions": [],
    }
    first = ReplayArchive(tmp_path)
    first.record("snapshot", snapshot, base)
    first.record("tick", {"symbol": "NFO:NIFTY_CE"}, base + timedelta(seconds=1))
    first.close()
    second = ReplayArchive(tmp_path)
    second.record("snapshot", snapshot, base + timedelta(seconds=2))
    second.record("tick", {"symbol": "NFO:NIFTY_CE"}, base + timedelta(seconds=3))
    second.close()

    original = tmp_path / "2026-10-21.jsonl"
    with pytest.raises(ValueError, match="replay_multiple_runs_unrestored_state"):
        validate_replay_epochs(original)
    parts = materialize_replay_epochs(original, tmp_path / "segments")

    assert len(parts) == 2
    assert [validate_replay_epochs(part)[0] for part in parts] == [2, 2]
    assert [len(list(iter_session(part))) for part in parts] == [2, 2]
    assert original.read_text() != parts[0].read_text()
    import json

    sidecar = next(parts[0].parent.glob("*.integrity.json"))
    assert json.loads(sidecar.read_text())["derived_epoch"] is True


def test_epoch_partition_does_not_invent_state_after_restart(tmp_path):
    from datetime import datetime, timedelta

    from nifty_scalper_bot.backtesting.runtime_research import (
        materialize_replay_epochs,
        validate_replay_epochs,
    )
    from nifty_scalper_bot.storage.replay_archive import ReplayArchive

    base = datetime.fromisoformat("2026-10-21T10:00:00+05:30")
    snapshot = {
        "instruments": [{"instrument_token": 123}],
        "effective_settings": {"risk": {"enabled": True}},
        "runner_config": {"mode": "test"},
        "basket": {"selected_ce": "NFO:NIFTY_CE"},
        "initial_balance": 100000,
        "initial_positions": [],
    }
    first = ReplayArchive(tmp_path)
    first.record("snapshot", snapshot, base)
    first.close()
    second = ReplayArchive(tmp_path)
    second.record(
        "snapshot",
        {**snapshot, "initial_positions": [{"quantity": 65}]},
        base + timedelta(seconds=2),
    )
    second.close()

    parts = materialize_replay_epochs(
        tmp_path / "2026-10-21.jsonl", tmp_path / "segments"
    )
    assert validate_replay_epochs(parts[0])[0] == 1
    with pytest.raises(ValueError, match="replay_initial_positions_unsupported"):
        validate_replay_epochs(parts[1])


def test_epoch_partition_refuses_missing_original_integrity(tmp_path):
    from datetime import datetime, timedelta

    from nifty_scalper_bot.backtesting.runtime_research import materialize_replay_epochs
    from nifty_scalper_bot.storage.replay_archive import ReplayArchive

    base = datetime.fromisoformat("2026-10-21T10:00:00+05:30")
    snapshot = {
        "instruments": [{"instrument_token": 123}],
        "effective_settings": {"risk": {"enabled": True}},
        "runner_config": {"mode": "test"},
        "basket": {"selected_ce": "NFO:NIFTY_CE"},
        "initial_balance": 100000,
        "initial_positions": [],
    }
    first = ReplayArchive(tmp_path)
    first.record("snapshot", snapshot, base)
    first.close()
    second = ReplayArchive(tmp_path)
    second.record("snapshot", snapshot, base + timedelta(seconds=1))
    second.close()
    (tmp_path / f"{second.run_id}.integrity.json").unlink()

    with pytest.raises(ValueError, match="replay_capture_integrity_missing"):
        materialize_replay_epochs(
            tmp_path / "2026-10-21.jsonl", tmp_path / "segments"
        )

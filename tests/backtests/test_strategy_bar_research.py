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


def test_component_replay_marks_intrabar_order_ambiguous_and_stresses_stop_first(
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
    result = report["scenarios"][0]["strategies"]["ORBPro"]
    assert result["trades"] == []
    assert result["data_quality"]["unresolved_exit_count"] == 2
    unresolved = result["unresolved_trades"]
    assert unresolved[0]["entry_time"] == (first + timedelta(minutes=1)).isoformat()
    assert all(
        trade["unresolved_reason"] == "ambiguous_intrabar_stop_target"
        for trade in unresolved
    )
    stress = result["worst_case_stress_trades"]
    assert len(stress) == 2
    assert stress[0]["exit_reason"] == "ambiguous_stop_first_stress"
    assert stress[0]["exit_price"] == 95
    assert stress[0]["fees"] > 0
    assert stress[0]["net_pnl"] < stress[0]["gross_pnl"]
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
    result = report["scenarios"][0]["strategies"]["SMC"]
    assert result["trades"] == []
    assert [trade["symbol"] for trade in result["unresolved_trades"]] == [
        "NFO:NIFTY26O0622400PE"
    ]
    assert result["no_vote_reasons"]["next_minute_unavailable"] == 1


def test_vwap_component_receives_underlying_futures_context(tmp_path, monkeypatch):
    from nifty_scalper_bot.backtesting.strategy_research import run_archived_research

    archive(tmp_path)
    observed = []

    class Strategy:
        name = "VWAPPro"
        config = SimpleNamespace(enabled=True)

        def generate_signal(self, symbol, indicators, current_price, position=None):
            observed.append(
                {
                    key: indicators.get(key)
                    for key in (
                        "underlying_direction_bias",
                        "direction_bias",
                        "underlying_direction_confidence",
                        "futures_vwap_slope",
                        "futures_volume_ratio",
                        "context_fresh",
                    )
                }
            )
            return None

        @property
        def evaluation_health(self):
            return {"healthy": True}

        def notify_entry_accepted(self, side, *, setup_id=None):
            pass

    context = {
        "underlying_direction_bias": "CE",
        "direction_bias": "CE",
        "underlying_direction_confidence": 0.91,
        "context_fresh": True,
        "futures_vwap_slope": 0.05,
        "futures_volume_ratio": 1.25,
    }
    monkeypatch.setattr(
        "nifty_scalper_bot.backtesting.strategy_research.build_elite_strategies",
        lambda settings, engine: [Strategy()],
    )
    monkeypatch.setattr(
        "nifty_scalper_bot.backtesting.strategy_research._research_orb_structural_context",
        lambda *args, **kwargs: dict(context),
    )

    run_archived_research(tmp_path, slippage_bps=0)

    assert observed
    assert all(row == context for row in observed)


def test_scenario_computes_shared_underlying_context_once_per_timestamp(
    tmp_path, monkeypatch
):
    from nifty_scalper_bot.backtesting.strategy_research import _scenario, load_archive

    archive(tmp_path)
    calls = []

    class Strategy:
        name = "VWAPPro"

        def generate_signal(self, symbol, indicators, current_price, position=None):
            return None

        @property
        def evaluation_health(self):
            return {"healthy": True}

    monkeypatch.setattr(
        "nifty_scalper_bot.backtesting.strategy_research.build_elite_strategies",
        lambda settings, engine: [Strategy()],
    )

    def context(*args, **kwargs):
        calls.append(kwargs.get("futures_symbol"))
        return {
            "underlying_direction_bias": "CE",
            "direction_bias": "CE",
            "underlying_direction_confidence": 0.8,
            "context_fresh": True,
            "futures_vwap_slope": 0.1,
            "futures_volume_ratio": 1.0,
        }

    monkeypatch.setattr(
        "nifty_scalper_bot.backtesting.strategy_research._research_orb_structural_context",
        context,
    )
    histories, instruments, _ = load_archive(tmp_path)

    _scenario(
        histories,
        instruments,
        None,
        0,
        components={"VWAPPro"},
    )

    assert len(calls) == 4


def test_research_structural_context_includes_futures_volume_ratio(tmp_path):
    from nifty_scalper_bot.backtesting.strategy_research import (
        IndicatorEngine,
        _research_orb_structural_context,
        load_archive,
    )

    first = archive(tmp_path)
    histories, _, _ = load_archive(tmp_path)
    engine = IndicatorEngine()
    for minute in range(4):
        timestamp = first + timedelta(minutes=minute)
        for symbol in ("NSE:NIFTY 50", "NFO:NIFTY26OCTFUT"):
            engine.ingest_historical_bar(symbol, histories[symbol][timestamp])

    context = _research_orb_structural_context(
        engine,
        spot_symbol="NSE:NIFTY 50",
        futures_symbol="NFO:NIFTY26OCTFUT",
    )

    assert context["futures_volume_ratio"] == pytest.approx(1.0)
    assert context["context_age_seconds"] == 0.0


def test_vwap_research_geometry_uses_structural_metadata_proxy() -> None:
    from nifty_scalper_bot.backtesting.strategy_research import (
        _resolve_research_signal_geometry,
    )

    signal = SimpleNamespace(
        stop_loss=None,
        take_profit=None,
        metadata={
            "setup_invalidation_premium": 95.0,
            "premium_target_rr": 2.0,
        },
    )

    stop, target, source = _resolve_research_signal_geometry(signal, 100.0)

    assert stop == 95.0
    assert target == 110.0
    assert source == "strategy_metadata_proxy"


def test_vwap_research_preserves_subtype_and_emits_setup_metrics(tmp_path, monkeypatch):
    from nifty_scalper_bot.backtesting.strategy_research import _scenario, load_archive

    archive(tmp_path)
    for path in (tmp_path / "candles").glob("*.json"):
        payload = json.loads(path.read_text())
        if payload["instrument"]["instrument_type"] in {"CE", "PE"}:
            payload["candles"][1][2] = 112.0
            payload["candles"][1][3] = 99.0
            payload["candles"][1][4] = 105.0
            path.write_text(json.dumps(payload))

    class Strategy:
        name = "VWAPPro"

        def generate_signal(self, symbol, indicators, current_price, position=None):
            if indicators["history_count"] == 1:
                return SimpleNamespace(
                    action="BUY",
                    stop_loss=None,
                    take_profit=None,
                    metadata={
                        "setup_id": f"vwap:{symbol}:1",
                        "setup_name": "vwap_reclaim",
                        "setup_type": "vwap_reclaim",
                        "signal_family": "reclaim_structure",
                        "setup_invalidation_premium": 95.0,
                        "premium_target_rr": 2.0,
                        "vwap_distance_atr": 0.4,
                        "futures_vwap_slope_bps": 2.5,
                        "option_volume_ratio": 0.9,
                        "futures_volume_ratio": 1.2,
                        "volume_confirmation_source": "option_and_futures",
                    },
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
    histories, instruments, _ = load_archive(tmp_path)
    result = _scenario(
        histories,
        instruments,
        None,
        0,
        components={"VWAPPro"},
    )[
        "strategies"
    ]["VWAPPro"]

    assert len(result["trades"]) == 2
    assert {trade["setup_name"] for trade in result["trades"]} == {"vwap_reclaim"}
    assert {trade["research_geometry_source"] for trade in result["trades"]} == {
        "strategy_metadata_proxy"
    }
    assert all(trade["vwap_distance_atr"] == 0.4 for trade in result["trades"])
    assert result["setup_metrics"]["vwap_reclaim"]["metrics"]["trade_count"] == 2
    assert result["setup_metrics"]["vwap_reclaim"]["metrics"]["net_pnl"] > 0


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


def test_vwap_comparison_is_bounded_restores_environment_and_never_auto_promotes(
    tmp_path, monkeypatch
):
    import os

    from nifty_scalper_bot.backtesting.strategy_research import run_vwap_comparison

    archive(tmp_path)
    observed: list[dict[str, str | None]] = []

    def replay(*args, **kwargs):
        observed.append(
            {
                "distance": os.environ.get("VWAP_TREND_QUALITY_MAX_DISTANCE_ATR"),
                "slope": os.environ.get("VWAP_FUTURES_SLOPE_MIN_BPS"),
                "futures_rvol": os.environ.get("VWAP_FUTURES_VOLUME_MIN_RATIO"),
                "penetration": os.environ.get("VWAP_ALLOW_PENETRATION_ONLY_ENTRY"),
            }
        )
        return {
            "strategies": {
                "VWAPPro": {
                    "metrics": {"trade_count": 0, "expectancy": None},
                    "development_metrics": {"trade_count": 0, "expectancy": None},
                    "data_quality": {"development_unresolved_exit_count": 0},
                    "trades": [],
                    "setup_metrics": {},
                }
            }
        }

    monkeypatch.setattr(
        "nifty_scalper_bot.backtesting.strategy_research._scenario",
        replay,
    )
    monkeypatch.setenv("VWAP_FUTURES_SLOPE_MIN_BPS", "1.25")
    before = dict(os.environ)

    report = run_vwap_comparison(tmp_path)

    assert dict(os.environ) == before
    assert report["candidate_count"] == 7
    assert report["slippage_scenario_count"] == 3
    assert len(observed) == 21
    slippages = {
        row["slippage_bps_per_side"] for row in report["candidates"][0]["scenarios"]
    }
    assert slippages == {10.0, 25.0, 50.0}
    assert report["selection"]["promotion_eligible"] is False
    assert report["selection"]["selected_for_live"] is None
    assert report["selection"]["development_ranking"] == []


def test_orb_comparison_uses_compact_orb_replay_for_all_fixed_scenarios(
    tmp_path, monkeypatch
):
    from nifty_scalper_bot.backtesting.strategy_research import run_orb_comparison

    archive(tmp_path)
    calls = []

    def replay(*args, **kwargs):
        calls.append(dict(kwargs))
        return {
            "strategies": {
                "ORBPro": {
                    "metrics": {"trade_count": 0, "expectancy": None},
                    "development_metrics": {"trade_count": 0, "expectancy": None},
                    "data_quality": {"development_unresolved_exit_count": 0},
                    "trades": [],
                    "setup_metrics": {},
                }
            }
        }

    monkeypatch.setattr(
        "nifty_scalper_bot.backtesting.strategy_research._scenario",
        replay,
    )

    report = run_orb_comparison(tmp_path)

    assert report["candidate_count"] == 9
    assert report["slippage_scenario_count"] == 3
    assert len(calls) == 27
    assert all(call["components"] == {"ORBPro"} for call in calls)
    assert all(call["compact_orb_context"] is True for call in calls)


def test_bounded_comparison_restores_environment_and_never_promotes_small_sample(
    tmp_path, monkeypatch
):
    import os

    from nifty_scalper_bot.backtesting.strategy_research import run_orb_comparison

    archive(tmp_path)
    monkeypatch.setenv("ORB_TARGET_RR", "1.9")
    before = dict(os.environ)
    report = run_orb_comparison(tmp_path)
    assert dict(os.environ) == before
    assert len(report["candidates"]) == 9
    assert report["selection"]["promotion_eligible"] is False
    assert report["selection"]["selected_for_live"] is None
    assert report["retrospective_check_is_untouched"] is False
    assert {
        row["slippage_bps_per_side"] for row in report["candidates"][0]["scenarios"]
    } == {10, 25, 50}
    assert all(
        row["development_metrics"]["trade_count"] == 0
        for candidate in report["candidates"]
        for row in candidate["scenarios"]
    )


def test_cost_filter_uses_slipped_fill_and_does_not_widen_stop(tmp_path, monkeypatch):
    from nifty_scalper_bot.backtesting.strategy_research import _scenario, load_archive

    archive(tmp_path)

    class Strategy:
        name = "ORBPro"

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
            raise AssertionError("cost-rejected setup must not be accepted")

    monkeypatch.setattr(
        "nifty_scalper_bot.backtesting.strategy_research.build_elite_strategies",
        lambda settings, engine: [Strategy()],
    )
    histories, instruments, _ = load_archive(tmp_path)
    result = _scenario(
        histories, instruments, None, 10, components={"ORBPro"}, minimum_net_rr=1.5
    )
    orb = result["strategies"]["ORBPro"]
    assert orb["trades"] == []
    assert orb["no_vote_reasons"]["cost_net_rr_rejected"] == 2


def test_opening_relative_volume_requires_prior_complete_sessions_only():
    from nifty_scalper_bot.backtesting.strategy_research import opening_relative_volume

    first = datetime.fromisoformat("2026-09-01T09:15:00+05:30")
    history = {}
    for day in range(16):
        for minute in range(5):
            history[first + timedelta(days=day, minutes=minute)] = {
                "volume": 200 if day == 14 else 100
            }
    now = first + timedelta(days=14, minutes=5)
    assert opening_relative_volume(history, now, 5) == 2
    assert opening_relative_volume(history, now - timedelta(minutes=2), 5) is None
    del history[first + timedelta(minutes=1)]
    assert opening_relative_volume(history, now, 5) is None

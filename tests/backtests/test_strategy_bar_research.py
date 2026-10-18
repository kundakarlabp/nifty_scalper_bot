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


def test_opening_above_target_has_known_first_touch_and_costed_profit(
    tmp_path, monkeypatch
):
    """Positive round trip: next-bar fill, target at next open, exact net ledger."""
    from nifty_scalper_bot.backtesting.strategy_research import _scenario, load_archive

    first = archive(tmp_path)
    ce_path = tmp_path / "candles/CE.json"
    payload = json.loads(ce_path.read_text())
    # At 09:31 a filled trade has no stop/target touch. At 09:32 the
    # opening print is already beyond its resting limit target; the later
    # low crosses the stop but cannot precede that opening execution.
    payload["candles"][1][1:] = [100, 103, 97, 100, 1000]
    payload["candles"][2][1:] = [110, 112, 94, 106, 1000]
    ce_path.write_text(json.dumps(payload))

    class Strategy:
        name = "ORBPro"

        def generate_signal(self, symbol, indicators, current_price, position=None):
            if symbol.endswith("CE") and indicators["history_count"] == 1:
                return SimpleNamespace(
                    action="BUY", stop_loss=95.0, take_profit=105.0, metadata={}
                )
            return None

        @property
        def evaluation_health(self):
            return {"healthy": True}

        def notify_entry_accepted(self, side, *, setup_id=None):
            pass

    histories, instruments, _ = load_archive(tmp_path)
    from nifty_scalper_bot.config.settings import get_settings

    result = _scenario(
        histories,
        instruments,
        get_settings().elite,
        0,
        components={"ORBPro"},
        strategy_factory=lambda engine: [Strategy()],
    )["strategies"]["ORBPro"]

    assert result["entry_funnel"]["accepted_fills"] == 1
    assert result["data_quality"]["unresolved_exit_count"] == 0
    assert result["metrics"]["trade_count"] == 1
    trade = result["trades"][0]
    assert trade["entry_time"] == (first + timedelta(minutes=1)).isoformat()
    assert trade["exit_time"] == (first + timedelta(minutes=3)).isoformat()
    assert trade["entry_price"] == 100.0
    assert trade["exit_price"] == 105.0
    assert trade["exit_reason"] == "gap_target"
    assert trade["quantity"] == 75
    assert trade["gross_pnl"] == pytest.approx(375.0)
    assert trade["fees"] > 0
    assert trade["net_pnl"] == pytest.approx(375.0 - trade["fees"])
    assert result["metrics"]["net_pnl"] == pytest.approx(trade["net_pnl"])


def test_full_orb_research_uses_causal_regime_detector(tmp_path, monkeypatch):
    """The full baseline may not silently omit the regime used by compact ORB."""
    from nifty_scalper_bot.backtesting import strategy_research as research

    archive(tmp_path)
    evaluations = []
    regime_calls = []

    class Detector:
        def evaluate(self, symbol, data):
            regime_calls.append(symbol)

    class RegimeManager:
        def __init__(self, detector):
            self.detector = detector

        def get_latest_snapshot(self):
            return SimpleNamespace(regime="TREND")

    class Strategy:
        name = "ORBPro"

        def generate_signal(self, symbol, indicators, current_price, position=None):
            evaluations.append(indicators.get("regime"))
            return None

        @property
        def evaluation_health(self):
            return {"healthy": True}

    monkeypatch.setattr(research, "MarketRegimeDetector", Detector)
    monkeypatch.setattr(research, "MarketRegimeManager", RegimeManager)
    monkeypatch.setattr(
        research, "build_elite_strategies", lambda settings, engine: [Strategy()]
    )
    research.run_archived_research(tmp_path, slippage_bps=0)
    assert regime_calls
    assert evaluations
    assert all(value == "TREND" for value in evaluations)


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


@pytest.mark.parametrize("strategy_name", ["VWAPPro", "SMC"])
def test_directional_components_receive_underlying_futures_context(
    tmp_path, monkeypatch, strategy_name
):
    from nifty_scalper_bot.backtesting.strategy_research import run_archived_research

    archive(tmp_path)
    observed = []

    class Strategy:
        name = strategy_name
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


def test_compact_orb_context_diagnostics_count_market_minutes_not_option_bars(
    tmp_path, monkeypatch
):
    from nifty_scalper_bot.backtesting.strategy_research import _scenario, load_archive
    from nifty_scalper_bot.config.settings import get_settings

    archive(tmp_path)

    class Strategy:
        name = "ORBPro"

        def generate_signal(self, symbol, indicators, current_price, position=None):
            return None

        @property
        def evaluation_health(self):
            return {"healthy": True}

    monkeypatch.setattr(
        "nifty_scalper_bot.backtesting.strategy_research._research_orb_structural_context",
        lambda *args, **kwargs: {
            "underlying_direction_bias": "CE",
            "direction_bias": "CE",
            "research_direction_resolution": "spot_futures_agree",
            "context_fresh": True,
        },
    )
    histories, instruments, _ = load_archive(tmp_path)
    scenario = _scenario(
        histories,
        instruments,
        get_settings().elite,
        0,
        components={"ORBPro"},
        compact_orb_context=True,
        strategy_factory=lambda engine: [Strategy()],
    )
    result = scenario["strategies"]["ORBPro"]

    assert result["entry_funnel"]["evaluations"] == 8
    quality = result["data_quality"]
    assert quality["orb_replay_coverage"]["matched_market_minutes"] == 4
    assert quality["orb_direction_resolution_minutes"] == {"spot_futures_agree": 4}
    assert sum(scenario["orb_replay_regimes"].values()) == 4


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
                "trend_momentum": os.environ.get("VWAP_TREND_MOMENTUM_ENABLED"),
                "retest": os.environ.get("VWAP_TREND_REQUIRE_RETEST"),
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
    monkeypatch.setenv("VWAP_TREND_MOMENTUM_ENABLED", "true")
    monkeypatch.setenv("VWAP_TREND_REQUIRE_RETEST", "true")
    before = dict(os.environ)

    report = run_vwap_comparison(tmp_path)

    assert dict(os.environ) == before
    assert report["candidate_count"] == 9
    assert report["slippage_scenario_count"] == 3
    assert len(observed) == 27
    assert sum(row["trend_momentum"] == "true" for row in observed) == 6
    assert sum(row["retest"] == "true" for row in observed) == 3
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
    import os

    from nifty_scalper_bot.backtesting.strategy_research import run_orb_comparison

    archive(tmp_path)
    calls = []

    def replay(*args, **kwargs):
        calls.append(
            {
                **dict(kwargs),
                "range_transition": os.getenv("ORB_RESEARCH_ALLOW_RANGE_TRANSITION"),
                "strict_retest_wick": os.getenv("ORB_RESEARCH_STRICT_RETEST_WICK"),
            }
        )
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

    assert report["candidate_count"] == 11
    assert report["slippage_scenario_count"] == 3
    assert len(calls) == 33
    assert all(call["components"] == {"ORBPro"} for call in calls)
    assert all(call["compact_orb_context"] is True for call in calls)
    assert sum(call["range_transition"] == "true" for call in calls) == 3
    assert sum(call["range_transition"] == "false" for call in calls) == 30
    assert sum(call["strict_retest_wick"] == "true" for call in calls) == 3
    assert sum(call["strict_retest_wick"] == "false" for call in calls) == 30


def test_compact_orb_replay_passes_range_regime_without_vwap_trend_flag(
    tmp_path, monkeypatch
):
    from nifty_scalper_bot.backtesting import strategy_research
    from nifty_scalper_bot.config.settings import get_settings

    archive(tmp_path)
    monkeypatch.delenv("VWAP_TREND_MOMENTUM_ENABLED", raising=False)
    seen_regimes = []

    class Detector:
        def evaluate(self, *_args, **_kwargs):
            return None

    class Manager:
        def __init__(self, detector):
            self.detector = detector

        def get_latest_snapshot(self):
            return SimpleNamespace(regime="RANGE")

    class ProbeORB:
        name = "ORBPro"
        config = SimpleNamespace(enabled=True)
        last_no_vote_reason = "underlying_orb_not_ready"

        @property
        def evaluation_health(self):
            return {"healthy": True}

        def generate_signal(self, symbol, indicators, current_price, position=None):
            seen_regimes.append(indicators.get("regime"))
            return None

    monkeypatch.setattr(strategy_research, "MarketRegimeDetector", Detector)
    monkeypatch.setattr(strategy_research, "MarketRegimeManager", Manager)
    monkeypatch.setattr(
        strategy_research,
        "build_elite_strategies",
        lambda settings, engine: [ProbeORB()],
    )
    histories, instruments, _ = strategy_research.load_archive(tmp_path)

    result = strategy_research._scenario(
        histories,
        instruments,
        get_settings().elite,
        10.0,
        components={"ORBPro"},
        compact_orb_context=True,
    )

    assert seen_regimes and set(seen_regimes) == {"RANGE"}
    assert result["orb_replay_regimes"]["RANGE"] > 0


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
    assert len(report["candidates"]) == 11
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


@pytest.mark.parametrize(
    "instrument,spot,when,reason",
    [
        (
            {"expiry": "2026-10-13", "strike": 22400.0},
            22401.0,
            "2026-09-15T10:01:00+05:30",
            "archive_weekly_expiry_not_proven",
        ),
        (
            {"expiry": "2026-10-13", "strike": 22400.0},
            22401.0,
            "2026-10-13T10:01:00+05:30",
            None,
        ),
        (
            {"expiry": "2026-10-13", "strike": 23000.0},
            22401.0,
            "2026-10-12T10:01:00+05:30",
            "archive_strike_outside_atm_basket",
        ),
        (
            {"expiry": "2026-10-13"},
            22401.0,
            "2026-10-12T10:01:00+05:30",
            "archive_strike_missing",
        ),
        (
            {"expiry": "2026-10-06", "strike": 22400.0},
            22401.0,
            "2026-10-12T10:01:00+05:30",
            "archive_weekly_expiry_not_proven",
        ),
    ],
)
def test_archived_weekly_atm_proxy_eligibility(instrument, spot, when, reason):
    from nifty_scalper_bot.backtesting.strategy_research import (
        _archived_weekly_atm_rejection,
    )

    assert (
        _archived_weekly_atm_rejection(instrument, datetime.fromisoformat(when), spot)
        == reason
    )


def test_vwap_research_requests_weekly_atm_scope_for_every_candidate(
    tmp_path, monkeypatch
):
    from nifty_scalper_bot.backtesting.strategy_research import run_vwap_comparison

    archive(tmp_path)
    observed = []

    def replay(*args, **kwargs):
        observed.append(kwargs)
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
        "nifty_scalper_bot.backtesting.strategy_research._scenario", replay
    )
    run_vwap_comparison(tmp_path)
    assert len(observed) == 27
    assert all(call["require_archived_weekly_atm"] is True for call in observed)


@pytest.mark.parametrize("include_strikes", [False, True])
def test_research_weekly_atm_scope_applies_before_strategy_vote(
    tmp_path, monkeypatch, include_strikes
):
    from nifty_scalper_bot.backtesting.strategy_research import _scenario, load_archive
    from nifty_scalper_bot.config.settings import get_settings

    archive(tmp_path)
    if include_strikes:
        for path in (tmp_path / "candles").glob("*.json"):
            payload = json.loads(path.read_text())
            if payload["instrument"]["instrument_type"] in {"CE", "PE"}:
                payload["instrument"]["strike"] = 100.0
                path.write_text(json.dumps(payload))

    observations = []

    class Candidate:
        name = "VWAPPro"

        def generate_signal(self, symbol, indicators, current_price, position=None):
            observations.append(symbol)
            return None

        @property
        def evaluation_health(self):
            return {"healthy": True}

    histories, instruments, _ = load_archive(tmp_path)
    report = _scenario(
        histories,
        instruments,
        get_settings().elite,
        10.0,
        components={"VWAPPro"},
        require_archived_weekly_atm=True,
        strategy_factory=lambda engine: [Candidate()],
    )
    assert report["archive_weekly_atm_proxy_enabled"] is True
    result = report["strategies"]["VWAPPro"]
    if include_strikes:
        assert result["evaluations"] == 8
        assert len(observations) == 8
    else:
        assert result["evaluations"] == 0
        assert observations == []
        assert result["no_vote_reasons"]["archive_strike_missing"] == 8


def test_weekly_atm_holdout_uses_only_valid_option_sessions(tmp_path):
    from nifty_scalper_bot.backtesting.strategy_research import _scenario, load_archive
    from nifty_scalper_bot.config.settings import get_settings

    archive(tmp_path)
    for path in (tmp_path / "candles").glob("*.json"):
        payload = json.loads(path.read_text())
        if payload["instrument"]["instrument_type"] in {"CE", "PE"}:
            payload["instrument"]["strike"] = 100.0
        baseline = payload["candles"]
        extra_dates = [
            "2026-09-15",
            "2026-09-16",
            "2026-09-17",
            "2026-09-18",
            "2026-10-02",
        ]
        payload["candles"] = [
            [
                f"{day}{row[0][10:]}",
                *row[1:],
            ]
            for day in extra_dates
            for row in baseline
        ] + baseline
        path.write_text(json.dumps(payload))

    class NoEntry:
        name = "VWAPPro"

        def generate_signal(self, *args, **kwargs):
            return None

        @property
        def evaluation_health(self):
            return {"healthy": True}

    histories, instruments, _ = load_archive(tmp_path)
    kwargs = dict(
        histories=histories,
        instruments=instruments,
        settings=get_settings().elite,
        slippage_bps=10.0,
        components={"VWAPPro"},
        strategy_factory=lambda engine: [NoEntry()],
    )
    unfiltered = _scenario(**kwargs)["strategies"]["VWAPPro"]
    scoped_report = _scenario(**kwargs, require_archived_weekly_atm=True)
    scoped = scoped_report["strategies"]["VWAPPro"]

    assert unfiltered["chronological_holdout_start"] == "2026-10-01"
    assert scoped["chronological_holdout_start"] == "2026-10-02"
    assert scoped["data_quality"]["source_option_session_count"] == 6
    assert scoped["data_quality"]["eligible_session_count"] == 2
    assert scoped["data_quality"]["eligible_option_bar_count"] == 16
    assert scoped["data_quality"]["eligible_session_dates"] == [
        "2026-10-01",
        "2026-10-02",
    ]


def test_orb_structural_failures_are_attributed_without_changing_trade_admission(
    tmp_path, monkeypatch
):
    from nifty_scalper_bot.backtesting.strategy_research import (
        _scenario,
        load_archive,
    )

    archive(tmp_path)

    class Strategy:
        name = "ORBPro"
        last_no_vote_reason = "orb_structural_contract_not_passed"
        last_structural_failure_reasons = (
            "underlying_volume_confirmation",
            "futures_vwap_slope_alignment",
        )

        def generate_signal(self, symbol, indicators, current_price, position=None):
            return None

        @property
        def evaluation_health(self):
            return {"healthy": True}

    monkeypatch.setattr(
        "nifty_scalper_bot.backtesting.strategy_research.build_elite_strategies",
        lambda settings, engine: [Strategy()],
    )
    histories, instruments, _ = load_archive(tmp_path)
    scenario = _scenario(histories, instruments, None, 25, components={"ORBPro"})
    result = scenario["strategies"]["ORBPro"]

    assert result["metrics"]["trade_count"] == 0
    assert result["entry_funnel"]["signals_emitted"] == 0
    observations = result["evaluations"]
    structural_rejections = result["no_vote_reasons"][
        "orb_structural_contract_not_passed"
    ]
    assert structural_rejections == observations
    assert result["structural_gate_failures"] == {
        "underlying_volume_confirmation": observations,
        "futures_vwap_slope_alignment": observations,
    }


def test_smc_research_funnel_counts_unique_underlying_setups_not_option_checks(
    tmp_path, monkeypatch
):
    from nifty_scalper_bot.backtesting.strategy_research import _scenario, load_archive

    archive(tmp_path)
    candles = tmp_path / "candles"
    extra_ce = json.loads((candles / "CE.json").read_text())
    extra_ce["symbol"] = "NFO:NIFTY26O0622450CE"
    (candles / "CE_extra.json").write_text(json.dumps(extra_ce))

    class Strategy:
        name = "SMC"
        last_no_vote_reason = "smc_awaiting_confirmation"

        def __init__(self):
            self._events = {}

        def generate_signal(self, symbol, indicators, current_price, position=None):
            side = "CE" if symbol.endswith("CE") else "PE"
            owner = indicators["futures_symbol"]
            key = (owner, side)
            if key not in self._events:
                self._events[key] = {"sweep_ts": indicators["bar_timestamp"]}
            if indicators["history_count"] == 2:
                sweep_ts = self._events[key]["sweep_ts"]
                return SimpleNamespace(
                    action="BUY",
                    stop_loss=95,
                    take_profit=105,
                    metadata={
                        "setup_id": f"smcv2:{owner}:{side}:{sweep_ts.isoformat()}"
                    },
                )
            return None

        @property
        def evaluation_health(self):
            return {"healthy": True}

        def notify_entry_accepted(self, side, *, setup_id=None):
            pass

    histories, instruments, _ = load_archive(tmp_path)
    report = _scenario(
        histories,
        instruments,
        None,
        0,
        components={"SMC"},
        strategy_factory=lambda engine: [Strategy()],
    )
    smc = report["strategies"]["SMC"]
    funnel = smc["smc_unique_setup_funnel"]

    assert smc["entry_funnel"]["signals_emitted"] == 3
    assert smc["entry_funnel"]["accepted_fills"] == 2
    assert smc["no_vote_reasons"]["smc_setup_already_filled"] == 1
    assert funnel["armed_setups"] == 2
    assert funnel["signaled_setups"] == 2
    assert funnel["simulated_filled_setups"] == 2
    assert funnel["rejection_reasons_by_unique_setup"]["smc_awaiting_confirmation"] == 2
    assert smc["evaluations"] > funnel["armed_setups"]


def test_smc_research_counts_terminal_rejection_even_after_event_is_removed(
    tmp_path, monkeypatch
):
    from nifty_scalper_bot.backtesting.strategy_research import _scenario, load_archive

    archive(tmp_path)

    class Strategy:
        name = "SMC"
        last_no_vote_reason = "smc_awaiting_confirmation"

        def __init__(self):
            self._events = {}

        def generate_signal(self, symbol, indicators, current_price, position=None):
            side = "CE" if symbol.endswith("CE") else "PE"
            key = (indicators["futures_symbol"], side)
            if indicators["history_count"] == 1:
                self._events[key] = {"sweep_ts": indicators["bar_timestamp"]}
                self.last_no_vote_reason = "smc_awaiting_confirmation"
            elif indicators["history_count"] == 2:
                self._events.pop(key, None)
                self.last_no_vote_reason = "smc_sweep_invalidated"
            else:
                self.last_no_vote_reason = "underlying_no_liquidity_sweep"
            return None

        @property
        def evaluation_health(self):
            return {"healthy": True}

    histories, instruments, _ = load_archive(tmp_path)
    report = _scenario(
        histories,
        instruments,
        None,
        0,
        components={"SMC"},
        strategy_factory=lambda engine: [Strategy()],
    )
    smc = report["strategies"]["SMC"]
    reasons = smc["smc_unique_setup_funnel"]["rejection_reasons_by_unique_setup"]

    assert smc["smc_unique_setup_funnel"]["armed_setups"] == 2
    assert reasons["smc_awaiting_confirmation"] == 2
    assert reasons["smc_sweep_invalidated"] == 2


@pytest.mark.parametrize(
    ("invalid_geometry", "rejection"),
    [
        (False, "next_minute_unavailable"),
        (True, "invalid_signal_geometry"),
    ],
)
def test_smc_post_signal_rejection_counts_unique_setups_not_option_attempts(
    tmp_path, invalid_geometry, rejection
):
    from nifty_scalper_bot.backtesting.strategy_research import _scenario, load_archive

    archive(tmp_path, gap=True)
    candles = tmp_path / "candles"
    extra_ce = json.loads((candles / "CE.json").read_text())
    extra_ce["symbol"] = "NFO:NIFTY26O0622450CE"
    (candles / "CE_extra.json").write_text(json.dumps(extra_ce))

    class Strategy:
        name = "SMC"
        last_no_vote_reason = "underlying_no_liquidity_sweep"

        def generate_signal(self, symbol, indicators, current_price, position=None):
            if not symbol.endswith("CE") or indicators["history_count"] != 1:
                return None
            return SimpleNamespace(
                action="BUY",
                stop_loss=101 if invalid_geometry else 95,
                take_profit=105,
                metadata={"setup_id": "smcv2:shared_underlying:CE:2026-10-01T09:30"},
            )

        @property
        def evaluation_health(self):
            return {"healthy": True}

    histories, instruments, _ = load_archive(tmp_path)
    result = _scenario(
        histories,
        instruments,
        None,
        0,
        components={"SMC"},
        strategy_factory=lambda engine: [Strategy()],
    )["strategies"]["SMC"]

    assert result["entry_funnel"]["signals_emitted"] == 2
    assert result["no_vote_reasons"][rejection] == 2
    unique_rejections = result["smc_unique_setup_funnel"][
        "post_signal_rejections_by_unique_setup"
    ]
    assert unique_rejections[rejection] == 1
    assert result["smc_unique_setup_funnel"]["simulated_filled_setups"] == 0


@pytest.mark.parametrize(
    ("premium_stop_distance", "expected_stop", "expected_target"),
    [(8.0, 92.0, 116.0), (200.0, 40.0, 220.0)],
)
def test_smc_component_research_resolves_canonical_option_premium_geometry(
    premium_stop_distance, expected_stop, expected_target
):
    """An SMC vote carries premium-risk metadata without explicit price levels."""
    from nifty_scalper_bot.backtesting.strategy_research import (
        _resolve_research_signal_geometry,
    )
    from nifty_scalper_bot.strategies.signal_generator import Signal

    signal = Signal(
        action="BUY",
        symbol="NFO:NIFTY26O0622400CE",
        quantity=75,
        confidence=1.0,
        reason="SMC",
        stop_loss=None,
        take_profit=None,
        metadata={
            "strategy": "SMC",
            "source_domain": "underlying_price",
            "premium_stop_distance": premium_stop_distance,
            "premium_target_rr": 2.0,
        },
    )
    stop, target, source = _resolve_research_signal_geometry(signal, 100.0)

    assert stop == pytest.approx(expected_stop)
    assert target == pytest.approx(expected_target)
    assert source == "canonical_option_premium_geometry"


def test_smc_archived_component_signal_can_reach_execution_feasibility(
    tmp_path, monkeypatch
):
    """Missing direct SL/TP must not reject a valid SMC premium-risk contract."""
    from nifty_scalper_bot.backtesting.strategy_research import run_archived_research
    from nifty_scalper_bot.strategies.signal_generator import Signal

    archive(tmp_path)

    class Strategy:
        name = "SMC"
        config = SimpleNamespace(enabled=True)

        def generate_signal(self, symbol, indicators, current_price, position=None):
            if indicators["history_count"] != 1:
                return None
            return Signal(
                action="BUY",
                symbol=symbol,
                quantity=75,
                confidence=1.0,
                reason="SMC",
                stop_loss=None,
                take_profit=None,
                metadata={
                    "strategy": "SMC",
                    "premium_stop_distance": 8.0,
                    "premium_target_rr": 2.0,
                },
            )

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

    assert result["entry_funnel"]["signals_emitted"] == 2
    assert result["entry_funnel"]["geometry_valid"] == 2
    assert result["entry_funnel"]["accepted_fills"] == 2
    assert result["no_vote_reasons"].get("invalid_signal_geometry", 0) == 0


def test_smc_research_does_not_refill_an_already_accepted_underlying_setup(
    tmp_path, monkeypatch
):
    from nifty_scalper_bot.backtesting.strategy_research import _scenario, load_archive

    archive(tmp_path)

    class Strategy:
        name = "SMC"
        last_no_vote_reason = "underlying_no_liquidity_sweep"

        def generate_signal(self, symbol, indicators, current_price, position=None):
            if indicators["history_count"] not in {2, 3}:
                return None
            side = "CE" if symbol.endswith("CE") else "PE"
            return SimpleNamespace(
                action="BUY",
                stop_loss=95,
                take_profit=105,
                metadata={"setup_id": f"smcv2:shared_underlying:{side}:same_sweep"},
            )

        @property
        def evaluation_health(self):
            return {"healthy": True}

        def notify_entry_accepted(self, side, *, setup_id=None):
            pass

    histories, instruments, _ = load_archive(tmp_path)
    result = _scenario(
        histories,
        instruments,
        None,
        0,
        components={"SMC"},
        strategy_factory=lambda engine: [Strategy()],
    )["strategies"]["SMC"]

    assert result["entry_funnel"]["signals_emitted"] == 4
    assert result["entry_funnel"]["next_bar_intents"] == 2
    assert result["entry_funnel"]["accepted_fills"] == 2
    assert result["no_vote_reasons"]["smc_setup_already_filled"] == 2
    assert result["smc_unique_setup_funnel"]["simulated_filled_setups"] == 2


def test_smc_research_failed_candidate_does_not_consume_shared_setup(
    tmp_path, monkeypatch
):
    from nifty_scalper_bot.backtesting.strategy_research import _scenario, load_archive

    archive(tmp_path)
    candles = tmp_path / "candles"
    extra_ce = json.loads((candles / "CE.json").read_text())
    extra_ce["symbol"] = "NFO:NIFTY26O0622450CE"
    (candles / "CE_extra.json").write_text(json.dumps(extra_ce))
    original_ce = json.loads((candles / "CE.json").read_text())
    # history_count=2 emits on the second completed candle, so the next
    # candle is the executable third minute: only this candidate lacks trades.
    original_ce["candles"][2][-1] = 0
    (candles / "CE.json").write_text(json.dumps(original_ce))

    class Strategy:
        name = "SMC"
        last_no_vote_reason = "underlying_no_liquidity_sweep"

        def generate_signal(self, symbol, indicators, current_price, position=None):
            if indicators["history_count"] != 2:
                return None
            side = "CE" if symbol.endswith("CE") else "PE"
            return SimpleNamespace(
                action="BUY",
                stop_loss=95,
                take_profit=105,
                metadata={"setup_id": f"smcv2:shared_underlying:{side}:same_sweep"},
            )

        @property
        def evaluation_health(self):
            return {"healthy": True}

        def notify_entry_accepted(self, side, *, setup_id=None):
            pass

    histories, instruments, _ = load_archive(tmp_path)
    result = _scenario(
        histories,
        instruments,
        None,
        0,
        components={"SMC"},
        strict_liquidity=True,
        strategy_factory=lambda engine: [Strategy()],
    )["strategies"]["SMC"]

    assert result["entry_funnel"]["signals_emitted"] == 3, result["entry_funnel"]
    assert result["no_vote_reasons"].get("next_minute_has_no_trades", 0) == 1, (
        result["entry_funnel"],
        result["no_vote_reasons"],
    )
    assert result["entry_funnel"]["accepted_fills"] == 2
    assert result["smc_unique_setup_funnel"]["simulated_filled_setups"] == 2


def test_vwap_research_cannot_inherit_optional_finalized_bar_gate(
    tmp_path, monkeypatch
):
    """All research arms must share the documented candle-completion policy."""
    import os

    from nifty_scalper_bot.backtesting.strategy_research import run_vwap_comparison

    archive(tmp_path)
    monkeypatch.setenv("VWAP_REQUIRE_FINALIZED_SIGNAL_BAR", "true")
    observed = []

    def fake_scenario(*args, **kwargs):
        observed.append(os.getenv("VWAP_REQUIRE_FINALIZED_SIGNAL_BAR"))
        return {
            "strategies": {
                "VWAPPro": {
                    "development_metrics": {"trade_count": 0, "expectancy": None},
                    "holdout_metrics": {"trade_count": 0, "expectancy": None},
                    "data_quality": {
                        "development_unresolved_exit_count": 0,
                        "holdout_unresolved_exit_count": 0,
                    },
                }
            }
        }

    monkeypatch.setattr(
        "nifty_scalper_bot.backtesting.strategy_research._scenario",
        fake_scenario,
    )
    before = dict(os.environ)
    report = run_vwap_comparison(tmp_path)

    assert observed == ["false"] * 27
    assert dict(os.environ) == before
    assert report["baseline_environment"]["VWAP_REQUIRE_FINALIZED_SIGNAL_BAR"] == (
        "true"
    )


def test_vwap_research_reports_a_cost_matched_retest_ablation(tmp_path, monkeypatch):
    """Immediate-vs-retest comparison must hold premium RVOL and fill costs fixed."""
    import os

    from nifty_scalper_bot.backtesting.strategy_research import run_vwap_comparison

    archive(tmp_path)
    observed = []

    def fake_scenario(*args, **kwargs):
        momentum = os.getenv("VWAP_TREND_MOMENTUM_ENABLED") == "true"
        retest = os.getenv("VWAP_TREND_REQUIRE_RETEST") == "true"
        rvol = os.getenv("VWAP_TREND_MOMENTUM_OPTION_RVOL_MIN")
        observed.append((momentum, retest, rvol, args[3]))
        expectancy = (4.0 if retest else 2.0) if momentum else -1.0
        count = (32 if retest else 40) if momentum else 0
        return {
            "strategies": {
                "VWAPPro": {
                    "development_metrics": {
                        "trade_count": count,
                        "expectancy": expectancy,
                    },
                    "holdout_metrics": {"trade_count": 0, "expectancy": None},
                    "data_quality": {
                        "eligible_session_count": 10,
                        "development_eligible_session_count": 8,
                        "holdout_eligible_session_count": 2,
                        "development_unresolved_exit_count": 0,
                        "holdout_unresolved_exit_count": 0,
                    },
                }
            }
        }

    monkeypatch.setattr(
        "nifty_scalper_bot.backtesting.strategy_research._scenario",
        fake_scenario,
    )
    report = run_vwap_comparison(tmp_path)
    comparison = report["paired_comparisons"][0]
    assert comparison["control"] == "trend_premium_rvol_1_5"
    assert comparison["treatment"] == "trend_premium_vwap_retest"
    assert comparison["only_changed_rule"] == "VWAP_TREND_REQUIRE_RETEST"
    assert len(comparison["scenarios"]) == 3
    assert {
        (momentum, retest, rvol) for momentum, retest, rvol, _ in observed if momentum
    } == {(True, False, "1.5"), (True, True, "1.5")}
    for scenario in comparison["scenarios"]:
        assert scenario["development_trade_count_control"] == 40
        assert scenario["development_trade_count_treatment"] == 32
        assert scenario["development_expectancy_delta"] == 2.0
        assert scenario["development_sample_eligible"] is True
        assert scenario["holdout_expectancy_delta"] is None
    assert comparison["selected_for_live"] is None


@pytest.mark.parametrize(
    ("absolute_level", "slippage_bps"),
    [(False, 0.0), (False, 25.0), (True, 0.0)],
)
def test_smc_research_bracket_anchors_at_actual_simulated_fill(
    tmp_path, monkeypatch, absolute_level, slippage_bps
):
    """Distance stops follow fill drift, but absolute technical levels do not."""
    from nifty_scalper_bot.backtesting.strategy_research import run_archived_research
    from nifty_scalper_bot.strategies.signal_generator import Signal

    archive(tmp_path)
    for path in (tmp_path / "candles").glob("*.json"):
        payload = json.loads(path.read_text())
        if payload["instrument"]["instrument_type"] in {"CE", "PE"}:
            # The first completed candle closes at 100 and emits SMC.
            # The next candle opens at 110, then hits the anchored target.
            payload["candles"][1][1:5] = [110.0, 130.0, 109.0, 110.0]
            path.write_text(json.dumps(payload))

    class Strategy:
        name = "SMC"
        config = SimpleNamespace(enabled=True)

        def generate_signal(self, symbol, indicators, current_price, position=None):
            if indicators["history_count"] != 1:
                return None
            metadata = {"strategy": "SMC"}
            if absolute_level:
                metadata["bracket_anchor_mode"] = "absolute_level"
            else:
                metadata.update(
                    premium_stop_distance=8.0,
                    premium_target_rr=2.0,
                )
            return Signal(
                action="BUY",
                symbol=symbol,
                quantity=75,
                confidence=1.0,
                reason="SMC",
                stop_loss=95.0 if absolute_level else None,
                take_profit=125.0 if absolute_level else None,
                metadata=metadata,
            )

        @property
        def evaluation_health(self):
            return {"healthy": True}

        def notify_entry_accepted(self, side, *, setup_id=None):
            pass

    monkeypatch.setattr(
        "nifty_scalper_bot.backtesting.strategy_research.build_elite_strategies",
        lambda settings, engine: [Strategy()],
    )
    result = run_archived_research(tmp_path, slippage_bps=slippage_bps)
    smc = next(
        scenario["strategies"]["SMC"]
        for scenario in result["scenarios"]
        if scenario["slippage_bps_per_side"] == slippage_bps
    )

    assert smc["entry_funnel"]["accepted_fills"] == 2
    assert len(smc["trades"]) == 2
    fill = 110.0 * (1.0 + slippage_bps / 10000.0)
    for trade in smc["trades"]:
        assert trade["entry_price"] == pytest.approx(fill)
        if absolute_level:
            assert trade["initial_stop_loss"] == pytest.approx(95.0)
            assert trade["take_profit"] == pytest.approx(125.0)
            assert trade["research_geometry_source"] == "signal"
        else:
            assert trade["initial_stop_loss"] == pytest.approx(fill - 8.0)
            assert trade["take_profit"] == pytest.approx(fill + 16.0)
            assert trade["research_geometry_source"] == (
                "canonical_option_premium_geometry"
            )
        assert trade["exit_reason"] == "target"


def test_orb_compact_replay_exposes_missing_market_data_separately_from_strategy(
    tmp_path,
) -> None:
    """Counts mutually exclusive missing-input reasons per replay minute, not strike."""
    from nifty_scalper_bot.backtesting.strategy_research import _scenario, load_archive
    from nifty_scalper_bot.config.settings import get_settings

    first = archive(tmp_path)
    histories, instruments, _ = load_archive(tmp_path)
    futures = "NFO:NIFTY26OCTFUT"
    histories[futures].pop(first + timedelta(minutes=1))
    histories["NSE:NIFTY 50"].pop(first + timedelta(minutes=2))
    for symbol in ("NFO:NIFTY26O0622400CE", "NFO:NIFTY26O0622400PE"):
        histories[symbol].pop(first + timedelta(minutes=3))

    result = _scenario(
        histories,
        instruments,
        get_settings().elite,
        25.0,
        components={"ORBPro"},
        compact_orb_context=True,
    )["strategies"]["ORBPro"]
    coverage = result["data_quality"]["orb_replay_coverage"]

    assert coverage == {
        "candidate_minutes": 4,
        "missing_underlying_minutes": 2,
        "missing_futures_minutes": 1,
        "missing_spot_minutes": 1,
        "missing_option_minutes": 1,
        "matched_market_minutes": 1,
        "pre_opening_range_end_minutes": 0,
        "complete_opening_range_minutes": 0,
        "incomplete_opening_range_minutes": 1,
    }
    assert result["entry_funnel"]["signals_emitted"] == 0


def test_orb_compact_replay_range_coverage_uses_only_prior_completed_minutes(
    tmp_path,
) -> None:
    from nifty_scalper_bot.backtesting.strategy_research import _scenario, load_archive
    from nifty_scalper_bot.config.settings import get_settings

    first = archive(tmp_path)
    histories, instruments, _ = load_archive(tmp_path)
    for symbol in ("NFO:NIFTY26OCTFUT", "NSE:NIFTY 50"):
        for minute in range(15):
            timestamp = first - timedelta(minutes=15 - minute)
            histories[symbol][timestamp] = {
                "timestamp": timestamp,
                "open": 100.0,
                "high": 106.0,
                "low": 94.0,
                "close": 100.0,
                "volume": 1000.0,
                "is_complete": True,
                "is_provisional": False,
            }

    result = _scenario(
        histories,
        instruments,
        get_settings().elite,
        25.0,
        components={"ORBPro"},
        compact_orb_context=True,
    )["strategies"]["ORBPro"]
    coverage = result["data_quality"]["orb_replay_coverage"]
    assert coverage["candidate_minutes"] == 4
    assert coverage["matched_market_minutes"] == 4
    assert coverage["complete_opening_range_minutes"] == 4
    assert coverage["incomplete_opening_range_minutes"] == 0


def test_orb_compact_replay_coverage_is_independent_of_option_strike_count(
    tmp_path,
) -> None:
    from nifty_scalper_bot.backtesting.strategy_research import _scenario, load_archive
    from nifty_scalper_bot.config.settings import get_settings

    archive(tmp_path)
    histories, instruments, _ = load_archive(tmp_path)
    result = _scenario(
        histories,
        instruments,
        get_settings().elite,
        10.0,
        components={"ORBPro"},
        compact_orb_context=True,
    )["strategies"]["ORBPro"]

    coverage = result["data_quality"]["orb_replay_coverage"]
    assert coverage["candidate_minutes"] == 4
    assert coverage["matched_market_minutes"] == 4
    assert coverage["incomplete_opening_range_minutes"] == 4
    assert result["evaluations"] == 8


@pytest.mark.parametrize(
    ("defect", "reason"),
    [
        ("token_collision", "research_instrument_token_collision"),
        ("mismatched_symbol", "research_instrument_symbol_mismatch"),
        ("outer_token_mismatch", "research_instrument_token_mismatch"),
    ],
)
def test_research_archive_rejects_conflicting_broker_instrument_identity(
    tmp_path, defect, reason
):
    from nifty_scalper_bot.backtesting.strategy_research import load_archive

    archive(tmp_path)
    for token, path in enumerate(sorted((tmp_path / "candles").glob("*.json")), 101):
        payload = json.loads(path.read_text())
        payload["instrument"]["instrument_token"] = token
        symbol = payload["symbol"]
        payload["instrument"]["exchange"] = symbol.split(":", 1)[0]
        payload["instrument"]["tradingsymbol"] = symbol.split(":", 1)[1]
        path.write_text(json.dumps(payload))

    ce_path = tmp_path / "candles" / "CE.json"
    pe_path = tmp_path / "candles" / "PE.json"
    ce = json.loads(ce_path.read_text())
    pe = json.loads(pe_path.read_text())
    if defect == "token_collision":
        pe["instrument"]["instrument_token"] = ce["instrument"]["instrument_token"]
        pe_path.write_text(json.dumps(pe))
    elif defect == "mismatched_symbol":
        ce["instrument"]["tradingsymbol"] = pe["instrument"]["tradingsymbol"]
        ce_path.write_text(json.dumps(ce))
    else:
        ce["instrument_token"] = ce["instrument"]["instrument_token"] + 999
        ce_path.write_text(json.dumps(ce))

    with pytest.raises(ValueError, match=reason):
        load_archive(tmp_path)


def test_research_archive_allows_token_reuse_after_expiry_on_a_later_date(tmp_path):
    """Broker tokens are not lifetime identifiers for expired derivatives."""
    from nifty_scalper_bot.backtesting.strategy_research import load_archive

    archive(tmp_path)
    ce_path = tmp_path / "candles" / "CE.json"
    old = json.loads(ce_path.read_text())
    old["instrument"].update(
        exchange="NFO",
        tradingsymbol=old["symbol"].split(":", 1)[1],
        instrument_token=13579,
    )
    ce_path.write_text(json.dumps(old))

    later = json.loads(json.dumps(old))
    later["symbol"] = "NFO:NIFTY26O1322400CE"
    later["instrument"].update(
        expiry="2026-10-13",
        tradingsymbol="NIFTY26O1322400CE",
    )
    later["candles"] = [
        [
            (datetime.fromisoformat(row[0]) + timedelta(days=7)).isoformat(),
            *row[1:],
        ]
        for row in later["candles"]
    ]
    (tmp_path / "candles" / "later_CE.json").write_text(json.dumps(later))

    histories, instruments, _ = load_archive(tmp_path)
    assert old["symbol"] in instruments
    assert later["symbol"] in instruments
    assert set(histories[old["symbol"]]).isdisjoint(histories[later["symbol"]])


def test_vwap_retest_archive_coverage_distinguishes_causal_history_failures(tmp_path):
    """Count actual observed option bars, never synthetic historical trades."""
    from nifty_scalper_bot.backtesting.strategy_research import (
        _vwap_retest_archive_coverage,
        load_archive,
    )

    archive(tmp_path)
    for file in (tmp_path / "candles").glob("*.json"):
        payload = json.loads(file.read_text())
        kind = payload["instrument"]["instrument_type"]
        first = datetime.fromisoformat("2026-10-01T09:15:00+05:30")
        payload["candles"] = [
            [
                (first + timedelta(minutes=i)).isoformat(),
                100,
                101,
                99,
                100,
                0 if kind == "PE" and i == 17 else 100,
            ]
            for i in range(19)
            if not (kind == "CE" and i == 1) and not (kind == "FUT" and i == 16)
        ]
        if kind in {"CE", "PE"}:
            payload["instrument"]["strike"] = 100
        file.write_text(json.dumps(payload))

    histories, instruments, _ = load_archive(tmp_path)
    coverage = _vwap_retest_archive_coverage(histories, instruments)
    counts = coverage["counts"]

    assert counts["observed_option_bars"] == 8
    assert counts["missing_underlying_minutes"] == 2
    assert counts["weekly_atm_proxy_rejections"] == 0
    assert counts["zero_volume_option_bars"] == 1
    assert counts["retest_eligible_option_bars"] == 5
    assert counts["prior_vwap_ready_option_bars"] == 1
    assert counts["missing_prior_session_history_option_bars"] == 3
    assert counts["zero_volume_prior_option_bars"] == 1
    assert counts["missing_prior_bar_option_bars"] == 0
    assert coverage["ready_share_of_eligible"] == pytest.approx(0.2)
    assert sum(coverage["by_side"]["CE"].values()) > 0
    assert sum(coverage["by_side"]["PE"].values()) > 0
    assert coverage["historical_selection_verified"] is False


def test_vwap_comparison_exposes_archive_coverage_without_changing_scenario(
    tmp_path, monkeypatch
):
    from nifty_scalper_bot.backtesting import strategy_research

    archive(tmp_path)
    calls = []

    def fake_scenario(*args, **kwargs):
        calls.append((args[3], kwargs["components"]))
        return {
            "strategies": {
                "VWAPPro": {
                    "development_metrics": {"trade_count": 0, "expectancy": None},
                    "holdout_metrics": {"trade_count": 0, "expectancy": None},
                    "data_quality": {
                        "development_unresolved_exit_count": 0,
                        "holdout_unresolved_exit_count": 0,
                    },
                }
            }
        }

    monkeypatch.setattr(strategy_research, "_scenario", fake_scenario)
    report = strategy_research.run_vwap_comparison(tmp_path)

    assert len(calls) == 27
    assert report["retest_archive_coverage"]["counts"]["observed_option_bars"] == 8
    assert (
        report["retest_archive_coverage"]["counts"]["weekly_atm_proxy_rejections"] == 8
    )
    assert report["selection"]["promotion_eligible"] is False


@pytest.mark.parametrize(
    ("days", "development_days", "holdout_days", "expected_status"),
    [
        (2, 1, 1, "insufficient_sessions"),
        (10, 8, 2, "descriptive_only"),
    ],
)
def test_vwap_research_blocks_development_ranking_on_sparse_session_coverage(
    tmp_path, monkeypatch, days, development_days, holdout_days, expected_status
):
    """Many modeled trades on one day are not an independent historical sample."""
    from nifty_scalper_bot.backtesting import strategy_research

    archive(tmp_path)

    def simulated_scenario(*args, **kwargs):
        return {
            "strategies": {
                "VWAPPro": {
                    "development_metrics": {"trade_count": 40, "expectancy": 1.5},
                    "holdout_metrics": {"trade_count": 5, "expectancy": 0.5},
                    "data_quality": {
                        "eligible_session_count": days,
                        "development_eligible_session_count": development_days,
                        "holdout_eligible_session_count": holdout_days,
                        "eligible_option_bar_count": 100,
                        "development_unresolved_exit_count": 0,
                        "holdout_unresolved_exit_count": 0,
                    },
                }
            }
        }

    monkeypatch.setattr(strategy_research, "_scenario", simulated_scenario)
    report = strategy_research.run_vwap_comparison(tmp_path)
    sample = report["sample_sufficiency"]

    assert sample["status"] == expected_status
    assert sample["eligible_session_count"] == days
    assert sample["development_eligible_session_count"] == development_days
    assert sample["holdout_eligible_session_count"] == holdout_days
    assert sample["minimum_total_sessions"] == 10
    assert sample["minimum_development_sessions"] == 8
    assert sample["minimum_holdout_sessions"] == 2
    if days < 10:
        assert "INSUFFICIENT_ELIGIBLE_SESSIONS" in sample["reason_codes"]
        assert not report["paired_comparisons"][0]["scenarios"][0][
            "development_sample_eligible"
        ]
        assert (
            report["paired_comparisons"][0]["scenarios"][0][
                "development_expectancy_delta"
            ]
            is None
        )
        assert report["selection"]["development_ranking"] == []
        assert report["selection"]["best_observed_research_candidate"] is None
    else:
        assert sample["reason_codes"] == []
        assert report["selection"]["development_ranking"]
    assert report["selection"]["promotion_eligible"] is False


def test_vwap_research_identifies_no_trades_without_zero_profitability_claim(
    tmp_path, monkeypatch
):
    from nifty_scalper_bot.backtesting import strategy_research

    archive(tmp_path)

    def simulated_scenario(*args, **kwargs):
        return {
            "strategies": {
                "VWAPPro": {
                    "development_metrics": {"trade_count": 0, "expectancy": None},
                    "holdout_metrics": {"trade_count": 0, "expectancy": None},
                    "data_quality": {
                        "eligible_session_count": 2,
                        "development_eligible_session_count": 1,
                        "holdout_eligible_session_count": 1,
                        "eligible_option_bar_count": 3300,
                        "development_unresolved_exit_count": 0,
                        "holdout_unresolved_exit_count": 0,
                    },
                }
            }
        }

    monkeypatch.setattr(strategy_research, "_scenario", simulated_scenario)
    report = strategy_research.run_vwap_comparison(tmp_path)
    assert report["sample_sufficiency"]["max_development_resolved_trades"] == 0
    assert (
        "NO_RESOLVED_DEVELOPMENT_TRADES" in report["sample_sufficiency"]["reason_codes"]
    )
    assert report["selection"]["development_ranking"] == []
    assert report["selection"]["selected_for_live"] is None


def test_orb_research_attributes_direction_once_per_replay_minute(
    tmp_path, monkeypatch
) -> None:
    """Option-strike fan-out must not multiply underlying direction evidence."""
    from nifty_scalper_bot.backtesting import strategy_research
    from nifty_scalper_bot.config.settings import get_settings

    archive(tmp_path)
    observations = []

    def prepared_context(engine, *, spot_symbol, futures_symbol):
        del engine, spot_symbol, futures_symbol
        observations.append(True)
        return {
            "research_direction_resolution": "spot_futures_conflict",
            "underlying_direction_bias": None,
            "direction_bias": None,
            "context_fresh": True,
            "futures_vwap_slope": 0.25,
            "futures_volume_ratio": 1.5,
        }

    monkeypatch.setattr(
        strategy_research,
        "_research_orb_structural_context",
        prepared_context,
    )
    histories, instruments, _ = strategy_research.load_archive(tmp_path)
    data = strategy_research._scenario(
        histories,
        instruments,
        get_settings().elite,
        10.0,
        components={"ORBPro"},
        compact_orb_context=True,
    )["strategies"]["ORBPro"]["data_quality"]

    assert len(observations) == 4
    assert data["orb_direction_resolution_minutes"] == {"spot_futures_conflict": 4}
    assert data["orb_replay_coverage"]["matched_market_minutes"] == 4
    assert data["eligible_option_bar_count"] == 8
    assert data["historical_executable_quote_fidelity"] == "ohlcv_only"


def test_orb_weekly_atm_proximity_reports_rejected_historical_basket_without_filter(
    tmp_path,
) -> None:
    from nifty_scalper_bot.backtesting.strategy_research import _scenario, load_archive
    from nifty_scalper_bot.config.settings import get_settings

    archive(tmp_path)
    histories, instruments, _ = load_archive(tmp_path)
    ce = "NFO:NIFTY26O0622400CE"
    pe = "NFO:NIFTY26O0622400PE"
    instruments[ce]["strike"] = 100
    instruments[pe]["strike"] = 350
    data = _scenario(
        histories,
        instruments,
        get_settings().elite,
        10.0,
        components={"ORBPro"},
        compact_orb_context=True,
    )["strategies"]["ORBPro"]["data_quality"]

    proxy = data["orb_weekly_atm_proximity_proxy"]
    assert proxy["candidate_option_bars"] == 8
    assert proxy["eligible_option_bars"] == 4
    assert proxy["eligible_minutes"] == 4
    assert proxy["rejection_reasons"] == {"archive_strike_outside_atm_basket": 4}
    assert data["eligible_option_bar_count"] == 8
    assert data["historical_executable_quote_fidelity"] == "ohlcv_only"


def test_orb_weekly_atm_proxy_does_not_infer_missing_expiry_or_strike(
    tmp_path,
) -> None:
    from nifty_scalper_bot.backtesting.strategy_research import _scenario, load_archive
    from nifty_scalper_bot.config.settings import get_settings

    archive(tmp_path)
    histories, instruments, _ = load_archive(tmp_path)
    result = _scenario(
        histories,
        instruments,
        get_settings().elite,
        25.0,
        components={"ORBPro"},
        compact_orb_context=True,
    )["strategies"]["ORBPro"]
    proxy = result["data_quality"]["orb_weekly_atm_proximity_proxy"]

    assert proxy["candidate_option_bars"] == 8
    assert proxy["eligible_option_bars"] == 0
    assert proxy["eligible_minutes"] == 0
    assert proxy["rejection_reasons"] == {"archive_strike_missing": 8}
    assert result["evaluations"] == 8


def test_vwap_research_isolates_all_inherited_variant_controls(tmp_path, monkeypatch):
    """The named research arms must not inherit another run's entry thresholds."""
    import os

    from nifty_scalper_bot.backtesting import strategy_research

    archive(tmp_path)
    injected = {
        "VWAP_TREND_QUALITY_MAX_DISTANCE_ATR": "9.0",
        "VWAP_FUTURES_SLOPE_MIN_BPS": "77",
        "VWAP_FUTURES_VOLUME_MIN_RATIO": "9",
        "VWAP_ALLOW_PENETRATION_ONLY_ENTRY": "true",
        "VWAP_TREND_MOMENTUM_OPTION_RVOL_MIN": "8",
    }
    for key, value in injected.items():
        monkeypatch.setenv(key, value)
    before = dict(os.environ)
    observations = []

    def fake_scenario(*args, **kwargs):
        observations.append({key: os.getenv(key) for key in injected})
        return {
            "strategies": {
                "VWAPPro": {
                    "development_metrics": {"trade_count": 0, "expectancy": None},
                    "holdout_metrics": {"trade_count": 0, "expectancy": None},
                    "data_quality": {
                        "development_unresolved_exit_count": 0,
                        "holdout_unresolved_exit_count": 0,
                    },
                }
            }
        }

    monkeypatch.setattr(strategy_research, "_scenario", fake_scenario)
    report = strategy_research.run_vwap_comparison(tmp_path)

    assert len(observations) == 27
    for index, candidate in enumerate(report["candidates"]):
        for observation in observations[index * 3 : (index + 1) * 3]:
            for key in injected:
                assert observation[key] == candidate["environment_overrides"].get(key)
    assert all(
        report["baseline_environment"][key] == value for key, value in injected.items()
    )
    assert dict(os.environ) == before


@pytest.mark.parametrize(
    ("days", "dev_days", "hold_days", "holdout_trades", "expected_eligible"),
    [
        (2, 1, 1, 15, False),
        (10, 8, 2, 1, False),
        (10, 8, 2, 10, True),
    ],
)
def test_vwap_holdout_delta_requires_session_and_trade_sufficiency(
    tmp_path, monkeypatch, days, dev_days, hold_days, holdout_trades, expected_eligible
):
    """One lucky holdout trade or two dates cannot establish a retest delta."""
    import os

    from nifty_scalper_bot.backtesting import strategy_research

    archive(tmp_path)

    def fake_scenario(*args, **kwargs):
        retest = os.getenv("VWAP_TREND_REQUIRE_RETEST") == "true"
        return {
            "strategies": {
                "VWAPPro": {
                    "development_metrics": {
                        "trade_count": 32 if retest else 40,
                        "expectancy": 4.0 if retest else 2.0,
                    },
                    "holdout_metrics": {
                        "trade_count": holdout_trades,
                        "expectancy": 4.0 if retest else 2.0,
                    },
                    "data_quality": {
                        "eligible_session_count": days,
                        "development_eligible_session_count": dev_days,
                        "holdout_eligible_session_count": hold_days,
                        "development_unresolved_exit_count": 0,
                        "holdout_unresolved_exit_count": 0,
                    },
                }
            }
        }

    monkeypatch.setattr(strategy_research, "_scenario", fake_scenario)
    report = strategy_research.run_vwap_comparison(tmp_path)
    comparison = report["paired_comparisons"][0]
    assert comparison["minimum_holdout_trades_per_arm"] == 10
    for scenario in comparison["scenarios"]:
        assert scenario["holdout_trade_count_control"] == holdout_trades
        assert scenario["holdout_trade_count_treatment"] == holdout_trades
        assert scenario["holdout_sample_eligible"] is expected_eligible
        assert scenario["holdout_expectancy_delta"] == (
            2.0 if expected_eligible else None
        )
    assert comparison["promotion_eligible"] is False

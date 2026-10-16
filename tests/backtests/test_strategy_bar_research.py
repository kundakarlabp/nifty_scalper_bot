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

    assert report["candidate_count"] == 10
    assert report["slippage_scenario_count"] == 3
    assert len(calls) == 30
    assert all(call["components"] == {"ORBPro"} for call in calls)
    assert all(call["compact_orb_context"] is True for call in calls)
    assert sum(call["range_transition"] == "true" for call in calls) == 3
    assert sum(call["range_transition"] == "false" for call in calls) == 27


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
    assert len(report["candidates"]) == 10
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
    assert smc["entry_funnel"]["accepted_fills"] == 3
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
    assert result["smc_unique_setup_funnel"][
        "post_signal_rejections_by_unique_setup"
    ][rejection] == 1
    assert result["smc_unique_setup_funnel"]["simulated_filled_setups"] == 0

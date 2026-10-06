import inspect

from nifty_scalper_bot.core.strategy_manager import signal_to_evidence
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    VWAPProStrategyConfig,
)
from nifty_scalper_bot.strategies.elite_strategies.vwap_pro import VWAPProStrategy


class _DummyEngine:
    pass


BAR_TS = 1_785_000_000.0


def _indicators(**updates):
    payload = {
        "vwap": 100.0,
        "atr": 5.0,
        "close": 103.0,
        "open": 100.0,
        "high": 104.0,
        "low": 99.0,
        "volume": 1000.0,
        "avg_volume": 900.0,
        "spread_pct": 0.5,
        "bid": 102.5,
        "ask": 103.0,
        "quote_depth_valid": True,
        "tradable_quote": True,
        "direction_bias": "CE",
        "underlying_direction_bias": "CE",
        "underlying_direction_confidence": 0.95,
        "context_age_seconds": 0.0,
        "context_fresh": True,
        "regime": "TREND_UP",
        "stale_data_used": False,
        "futures_vwap_slope": 1.0,
        "latest_bar_ts": BAR_TS,
    }
    payload.update(updates)
    return payload


def test_vwap_metadata_contract_side_and_structural_setup_present():
    strategy = VWAPProStrategy(VWAPProStrategyConfig(), _DummyEngine())
    signal = strategy._evaluate_signal(
        "NFO:NIFTY26FEB22500CE",
        _indicators(),
        101.0,
    )
    assert signal is not None
    metadata = signal.metadata
    assert metadata["contract_side"] == "CE"
    assert metadata["direction_bias"] == "CE"
    assert metadata["setup_pass"] is True
    assert "underlying_direction_alignment" in metadata["setup_reasons"]
    assert metadata["setup_id"].startswith("vwap:CE:")


def test_live_vwap_rejects_threshold_pass_from_small_noise(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    strategy = VWAPProStrategy(VWAPProStrategyConfig(), _DummyEngine())

    signal = strategy._evaluate_signal(
        "NFO:NIFTY26FEB22500CE",
        _indicators(
            close=100.05,
            open=100.04,
            high=100.10,
            low=99.90,
            volume=0.0,
            avg_volume=0.0,
            futures_volume_ratio=1.2,
        ),
        100.05,
    )

    assert signal is None
    assert strategy.last_no_vote_reason == "vwap_event_unconfirmed"


def test_live_vwap_accepts_meaningful_atr_penetration(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    strategy = VWAPProStrategy(VWAPProStrategyConfig(), _DummyEngine())

    signal = strategy._evaluate_signal(
        "NFO:NIFTY26FEB22500CE",
        _indicators(
            close=100.80,
            open=99.80,
            high=100.85,
            low=99.50,
            volume=0.0,
            avg_volume=0.0,
            futures_volume_ratio=1.2,
        ),
        100.80,
    )

    assert signal is not None
    assert signal.metadata["penetration_confirmed"] is True
    assert signal.metadata["vwap_event_confirmed"] is True


def test_relaxed_vwap_distance_requires_high_confidence_context(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    symbol = "NFO:NIFTY26FEB22500CE"
    setup = _indicators(
        atr=1.0,
        close=103.0,
        open=99.0,
        high=103.2,
        low=99.0,
        volume=1000.0,
        avg_volume=900.0,
        context_age_seconds=0.0,
        futures_vwap_slope=1.0,
    )

    low_confidence = VWAPProStrategy(VWAPProStrategyConfig(), _DummyEngine())
    rejected = low_confidence._evaluate_signal(
        symbol,
        {**setup, "underlying_direction_confidence": 0.50},
        103.0,
    )

    assert rejected is None
    assert low_confidence.last_no_vote_reason == "distance_outside_band"

    high_confidence = VWAPProStrategy(VWAPProStrategyConfig(), _DummyEngine())
    accepted = high_confidence._evaluate_signal(
        symbol,
        {**setup, "underlying_direction_confidence": 0.95},
        103.0,
    )

    assert accepted is not None
    assert accepted.metadata["vwap_distance_atr"] == 3.0
    assert accepted.metadata["vwap_strong_fresh_trend_context"] is True


def test_vwap_thesis_uses_stable_structural_id_until_closed_candle_reset(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    strategy = VWAPProStrategy(VWAPProStrategyConfig(), _DummyEngine())
    symbol = "NFO:NIFTY26FEB22500CE"

    first = strategy._evaluate_signal(symbol, _indicators(), 103.0)
    assert first is not None
    strategy.notify_entry_accepted("CE")

    same_thesis = strategy._evaluate_signal(
        symbol,
        _indicators(
            latest_bar_ts=BAR_TS + 60.0,
            close=104.0,
            open=103.8,
            high=104.2,
            low=103.5,
        ),
        104.0,
    )
    assert same_thesis is not None
    assert same_thesis.metadata["setup_id"] == first.metadata["setup_id"]

    assert (
        strategy._evaluate_signal(
            symbol,
            _indicators(
                latest_bar_ts=BAR_TS + 120.0,
                close=99.5,
                open=100.0,
                high=100.2,
                low=99.2,
            ),
            99.5,
        )
        is None
    )
    assert strategy.last_no_vote_reason == "vwap_thesis_reset"

    next_thesis = strategy._evaluate_signal(
        symbol,
        _indicators(
            latest_bar_ts=BAR_TS + 180.0,
            close=103.0,
            open=100.5,
            high=103.2,
            low=100.2,
        ),
        103.0,
    )
    assert next_thesis is not None
    assert next_thesis.metadata["setup_id"] != first.metadata["setup_id"]


def test_vwap_thesis_does_not_leak_across_same_side_contracts(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    strategy = VWAPProStrategy(VWAPProStrategyConfig(), _DummyEngine())

    assert (
        strategy._evaluate_signal(
            "NFO:NIFTY2690124100CE",
            _indicators(
                session_date="2026-09-01",
                close=99.5,
                open=100.2,
                high=100.4,
                low=99.2,
            ),
            99.5,
        )
        is None
    )
    assert strategy.last_no_vote_reason == "vwap_thesis_reset"

    rotated = strategy._evaluate_signal(
        "NFO:NIFTY2690124050CE",
        _indicators(
            session_date="2026-09-01",
            latest_bar_ts=BAR_TS + 60.0,
            close=103.0,
            open=102.0,
            high=103.2,
            low=101.8,
        ),
        103.0,
    )

    assert rotated is None
    assert strategy.last_no_vote_reason == "vwap_thesis_not_armed"


def test_vwap_thesis_does_not_leak_across_sessions(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    strategy = VWAPProStrategy(VWAPProStrategyConfig(), _DummyEngine())
    symbol = "NFO:NIFTY2690124100CE"

    assert (
        strategy._evaluate_signal(
            symbol,
            _indicators(
                session_date="2026-09-01",
                close=99.5,
                open=100.2,
                high=100.4,
                low=99.2,
            ),
            99.5,
        )
        is None
    )

    next_session = strategy._evaluate_signal(
        symbol,
        _indicators(
            session_date="2026-09-02",
            latest_bar_ts=BAR_TS + 86_400.0,
            close=103.0,
            open=102.0,
            high=103.2,
            low=101.8,
        ),
        103.0,
    )

    assert next_session is None
    assert strategy.last_no_vote_reason == "vwap_thesis_not_armed"


def test_real_vwap_emits_structural_evidence(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE", "true")
    strategy = VWAPProStrategy(VWAPProStrategyConfig(), _DummyEngine())
    indicators = _indicators()

    signal = strategy.generate_signal(
        "NFO:NIFTY26FEB22500CE",
        indicators,
        103.0,
    )

    assert signal is not None
    evidence = signal_to_evidence(signal, "VWAPPro")
    assert evidence.side == "CE"
    assert evidence.metadata["setup_pass"] is True
    assert evidence.metadata["requires_runner_execution_validation"] is True
    assert evidence.metadata["direction_bias"] == "CE"
    assert evidence.metadata["futures_slope_alignment"] is True
    assert evidence.metadata["volume_confirmation"] is True


def test_vwap_has_no_unreachable_early_trend_pullback_branch():
    init_source = inspect.getsource(VWAPProStrategy.__init__)
    evaluate_source = inspect.getsource(VWAPProStrategy._evaluate_signal)

    assert "_allow_early_trend_pullback" not in init_source
    assert "early_trend_pullback" not in evaluate_source
    assert "VWAP_PRO_ALLOW_EARLY_TREND_PULLBACK" not in init_source

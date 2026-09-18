from pathlib import Path

from nifty_scalper_bot.strategies.signal_quality import score_signal_quality


_RUNNER = Path("src/nifty_scalper_bot/strategies/runner.py")
_ORB = Path("src/nifty_scalper_bot/strategies/elite_strategies/orb_pro.py")
_SIGNAL_QUALITY = Path("src/nifty_scalper_bot/strategies/signal_quality.py")


def test_runner_has_one_final_quality_decision_owner() -> None:
    source = _RUNNER.read_text(encoding="utf-8")
    assert "quality.final_score < live_threshold" not in source
    assert 'metadata.get("direction_quality", quality_hint)' not in source
    assert 'quality_reject_reason = "final_score_below_live_threshold"' in source
    assert "if not quality.allowed:" in source


def test_signal_quality_has_no_dead_alternate_scoring_engine() -> None:
    source = _SIGNAL_QUALITY.read_text(encoding="utf-8")
    assert source.count("def score_signal_quality(") == 1
    assert "def compute_final_execution_score(" not in source
    assert "def compute_context_boost(" not in source
    assert "def context_boost_cap(" not in source
    assert "def rejection_cooldown(" not in source


def test_orb_pro_publishes_native_direction_evidence() -> None:
    source = _ORB.read_text(encoding="utf-8")
    assert '"direction_score": strategy_score' in source


def test_signal_quality_allowed_owns_threshold_and_direction(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    low_score = score_signal_quality(
        direction_score=7.0,
        strategy_score=7.0,
        option_score=7.0,
        data_score=7.0,
        rr_score=7.0,
        strategy_name="VWAPPro",
    )
    assert low_score.allowed is False
    assert "score_below_threshold" in low_score.reasons

    low_direction = score_signal_quality(
        direction_score=5.9,
        strategy_score=10.0,
        option_score=10.0,
        data_score=10.0,
        rr_score=10.0,
        strategy_name="VWAPPro",
    )
    assert low_direction.allowed is False
    assert "direction_below_minimum" in low_direction.reasons

    context_only = score_signal_quality(
        direction_score=10.0,
        strategy_score=10.0,
        option_score=10.0,
        data_score=10.0,
        rr_score=10.0,
        strategy_name="OrderFlow",
    )
    assert context_only.allowed is False
    assert "context_only_strategy" in context_only.reasons


def test_execution_quality_cannot_rescue_weak_directional_alpha(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    quality = score_signal_quality(
        direction_score=7.8,
        strategy_score=6.4,
        option_score=10.0,
        data_score=10.0,
        rr_score=10.0,
        strategy_name="VWAPPro",
    )
    assert quality.final_score > quality.components["threshold"]
    assert quality.components["alpha_score"] < quality.components["threshold"]
    assert quality.allowed is False
    assert "alpha_below_threshold" in quality.reasons


def test_strong_alpha_with_good_execution_remains_tradable(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    quality = score_signal_quality(
        direction_score=8.5,
        strategy_score=8.0,
        option_score=9.0,
        data_score=9.0,
        rr_score=8.0,
        strategy_name="VWAPPro",
    )
    assert quality.components["alpha_score"] >= quality.components["threshold"]
    assert quality.allowed is True


def test_vwap_runner_uses_independent_alpha_for_quality_and_confidence() -> None:
    source = _RUNNER.read_text(encoding="utf-8")
    assert 'metadata.get("independent_setup_score"' in source
    assert 'quality.components.get("alpha_score", quality.final_score)' in source
    assert '"alpha_score": alpha_score' in source

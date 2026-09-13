from pathlib import Path

from nifty_scalper_bot.strategies.signal_quality import score_signal_quality


_RUNNER = Path("src/nifty_scalper_bot/strategies/runner.py")
_ORB = Path("src/nifty_scalper_bot/strategies/elite_strategies/orb_pro.py")


def test_runner_has_one_final_quality_decision_owner() -> None:
    source = _RUNNER.read_text(encoding="utf-8")
    assert "quality.final_score < live_threshold" not in source
    assert 'metadata.get("direction_quality", quality_hint)' not in source
    assert 'quality_reject_reason = "final_score_below_live_threshold"' in source
    assert "if not quality.allowed:" in source


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

from __future__ import annotations

from nifty_scalper_bot.strategies.signal_quality import infer_option_side, score_signal_quality


def test_signal_quality_weighted_score_and_confidence_model(monkeypatch) -> None:
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    monkeypatch.setenv('SIGNAL_MIN_SCORE_LIVE', '8.0')
    result = score_signal_quality(
        direction_score=8.0,
        strategy_score=8.0,
        option_score=8.0,
        data_score=8.0,
        rr_score=8.0,
    )
    assert result.final_score == 8.0
    assert result.allowed is True


def test_signal_quality_live_threshold_blocks_low_score(monkeypatch) -> None:
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    monkeypatch.setenv('SIGNAL_MIN_SCORE_LIVE', '8.0')
    result = score_signal_quality(
        direction_score=6.0,
        strategy_score=7.0,
        option_score=7.0,
        data_score=7.0,
        rr_score=7.0,
    )
    assert result.allowed is False
    assert 'score_below_threshold' in result.reasons


def test_infer_option_side_detects_ce_and_pe() -> None:
    assert infer_option_side('NFO:NIFTY26APR23800PE', {}) == 'PE'
    assert infer_option_side('NFO:NIFTY26APR23800CE', {}) == 'CE'


def test_infer_option_side_unknown_uses_metadata() -> None:
    assert infer_option_side('NSE:NIFTY', {'direction_bias': 'PE'}) == 'PE'

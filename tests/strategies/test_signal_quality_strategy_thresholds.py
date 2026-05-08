from nifty_scalper_bot.strategies.signal_quality import score_signal_quality


def test_strategy_threshold_aliases_live(monkeypatch):
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    rsi = score_signal_quality(direction_score=8, strategy_score=8, option_score=8, data_score=8, rr_score=7, strategy_name='RSIDivergence')
    assert rsi.components['threshold'] == 7.6
    assert rsi.allowed
    unk = score_signal_quality(direction_score=8, strategy_score=8, option_score=8, data_score=8, rr_score=7, strategy_name='unknown')
    assert unk.components['threshold'] == 8.0
    assert not (unk.final_score < 8.0 and unk.allowed)
    ps = score_signal_quality(direction_score=7.5, strategy_score=7.5, option_score=7.5, data_score=7.5, rr_score=7.0, strategy_name='premium_momentum_squeeze')
    assert ps.components['threshold'] == 7.4
    assert ps.allowed

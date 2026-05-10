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
    smc = score_signal_quality(direction_score=7.2, strategy_score=7.2, option_score=7.2, data_score=7.2, rr_score=7.2, strategy_name='smc')
    assert smc.components['threshold'] == 7.0
    assert smc.allowed
    cpr = score_signal_quality(direction_score=7.3, strategy_score=7.3, option_score=7.3, data_score=7.3, rr_score=7.3, strategy_name='CPRBreakout')
    assert cpr.components['threshold'] == 7.2
    bb = score_signal_quality(direction_score=7.4, strategy_score=7.4, option_score=7.4, data_score=7.4, rr_score=7.4, strategy_name='BBSqueeze')
    assert bb.components['threshold'] == 7.3
    of = score_signal_quality(direction_score=7.2, strategy_score=7.2, option_score=7.2, data_score=7.2, rr_score=7.2, strategy_name='OrderFlow')
    assert of.components['threshold'] == 7.1
    orb = score_signal_quality(direction_score=7.5, strategy_score=7.5, option_score=7.5, data_score=7.5, rr_score=7.5, strategy_name='ORBPro')
    assert orb.components['threshold'] == 7.4

from nifty_scalper_bot.strategies.signal_quality import score_signal_quality


def test_strategy_name_threshold_present_live(monkeypatch):
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    q = score_signal_quality(direction_score=8, strategy_score=8, option_score=8, data_score=8, rr_score=8, strategy_name='premium_momentum_squeeze')
    assert q.components['strategy_name'] == 'premium_momentum_squeeze'
    assert q.components['threshold'] == 7.4

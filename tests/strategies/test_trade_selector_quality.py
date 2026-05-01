from nifty_scalper_bot.strategies.trade_selector import TradeCandidateSelector


def base(**kw):
    s = {'symbol': 'NFO:NIFTY123CE', 'side': 'CE', 'strike': 22000, 'ltp': 120, 'bid': 119, 'ask': 121, 'tick_age_s': 2, 'real_ticks_last_60s': 3, 'atr_option': 7}
    s.update(kw)
    return s


def test_stale_rejected():
    r = TradeCandidateSelector(quality_mode='strict').select_ranked_candidates(direction_bias='CE', atm_strike=22000, snapshots=[base(tick_age_s=11)])
    assert not r


def test_wide_spread_rejected():
    r = TradeCandidateSelector().select_ranked_candidates(direction_bias='CE', atm_strike=22000, snapshots=[base(bid=100, ask=130)])
    assert not r


def test_valid_selected_and_rr():
    r = TradeCandidateSelector().select_ranked_candidates(direction_bias='CE', atm_strike=22000, snapshots=[base()])
    assert r and r[0].rr and r[0].rr >= 1.5


def test_far_otm_rejected():
    r = TradeCandidateSelector().select_ranked_candidates(direction_bias='CE', atm_strike=22000, snapshots=[base(strike=22400)])
    assert not r

from nifty_scalper_bot.risk.cost_model import evaluate_net_reward_risk
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
    candidate = r[0]
    economics = evaluate_net_reward_risk(entry_price=candidate.entry_price, stop_price=candidate.stop_loss, target_price=candidate.target, quantity=65, half_spread=1.0)
    assert economics.allowed is True


def test_far_otm_rejected():
    r = TradeCandidateSelector().select_ranked_candidates(direction_bias='CE', atm_strike=22000, snapshots=[base(strike=22400)])
    assert not r


def test_tighter_spread_scores_higher_liquidity():
    selector = TradeCandidateSelector(max_option_spread_pct=1.0)
    tight = selector.select_ranked_candidates(direction_bias='CE', atm_strike=22000, snapshots=[base(symbol='NFO:NIFTYTIGHTCE', bid=119.88, ask=120.12)])
    wide = selector.select_ranked_candidates(direction_bias='CE', atm_strike=22000, snapshots=[base(symbol='NFO:NIFTYWIDECE', bid=119.64, ask=120.36)])
    assert tight and wide
    assert tight[0].liquidity_score > wide[0].liquidity_score
    assert tight[0].final_score > wide[0].final_score


def test_data_quality_uses_canonical_millisecond_age_over_stale_legacy_seconds():
    selector = TradeCandidateSelector()
    quality = selector.evaluate_data_quality(base(tick_age_s=99, tick_age_ms=250))
    assert quality.allowed is True
    assert quality.score == 10.0


def test_ranked_candidate_reuses_resolved_readiness_for_quality_score():
    selector = TradeCandidateSelector()
    ranked = selector.select_ranked_candidates(direction_bias='CE', atm_strike=22000, snapshots=[base(tick_age_s=99, tick_age_ms=250)])
    assert ranked
    assert ranked[0].tick_age_s == 0.25
    assert ranked[0].data_quality_score == 10.0


def test_data_quality_respects_effective_candidate_limits():
    selector = TradeCandidateSelector(max_tick_age_s=1.0, require_real_ticks_last_60s=3)
    quality = selector.evaluate_data_quality(base(tick_age_ms=900), max_age=1.0, min_ticks=3, max_spread=5.0)
    assert quality.allowed is True
    assert quality.score == 10.0

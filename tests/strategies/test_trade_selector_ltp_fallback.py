from __future__ import annotations

import logging

from nifty_scalper_bot.strategies.trade_selector import TradeCandidateSelector


def _snapshot() -> dict[str, object]:
    return {
        'symbol': 'NFO:NIFTY26MAY24050CE',
        'side': 'CE',
        'strike': 24050,
        'atm_strike': 24050,
        'ltp': 120.0,
        'bid': 0.0,
        'ask': 0.0,
        'tick_age_s': 1.0,
        'real_ticks_last_60s': 2,
        'ltp_only_fallback': True,
    }


def test_ltp_only_candidate_allowed_and_penalized(monkeypatch) -> None:
    monkeypatch.setenv('ALLOW_LTP_ONLY_CANDIDATE', 'true')
    selector = TradeCandidateSelector()
    ranked = selector.select_ranked_candidates(direction_bias='CE', atm_strike=24050, snapshots=[_snapshot()])
    assert len(ranked) == 1
    assert 'ltp_only_fallback' in ranked[0].reasons
    assert ranked[0].spread_pct is None


def test_ltp_only_candidate_rejected_when_disabled(monkeypatch) -> None:
    monkeypatch.setenv('ALLOW_LTP_ONLY_CANDIDATE', 'false')
    selector = TradeCandidateSelector()
    ranked = selector.select_ranked_candidates(direction_bias='CE', atm_strike=24050, snapshots=[_snapshot()])
    assert ranked == []


def test_candidate_summary_empty_info_throttled(caplog, monkeypatch) -> None:
    monkeypatch.setenv('ALLOW_LTP_ONLY_CANDIDATE', 'false')
    selector = TradeCandidateSelector()
    with caplog.at_level(logging.INFO):
        selector.select_ranked_candidates(direction_bias='CE', atm_strike=24050, snapshots=[_snapshot()])
        selector.select_ranked_candidates(direction_bias='CE', atm_strike=24050, snapshots=[_snapshot()])
    infos = [r for r in caplog.records if r.levelno == logging.INFO and 'CANDIDATE_SELECTION_SUMMARY' in r.message]
    assert len(infos) == 1


def test_min_option_premium_env_and_constructor_override(monkeypatch) -> None:
    monkeypatch.setenv('MIN_OPTION_PREMIUM', '55')
    selector = TradeCandidateSelector()
    assert selector.min_option_premium == 55.0
    override = TradeCandidateSelector(min_option_premium=45.0)
    assert override.min_option_premium == 45.0


def test_premium_reject_logs_actionable_fields(caplog) -> None:
    selector = TradeCandidateSelector(min_option_premium=200.0, max_option_premium=650.0)
    with caplog.at_level(logging.INFO):
        ranked = selector.select_ranked_candidates(direction_bias='CE', atm_strike=24050, snapshots=[_snapshot()])
    assert ranked == []
    assert any("CANDIDATE_REJECTED symbol=NFO:NIFTY26MAY24050CE reason=premium_out_of_range" in r.message for r in caplog.records)


def test_bad_and_good_candidates_select_good() -> None:
    selector = TradeCandidateSelector(min_option_premium=100.0, max_option_premium=650.0)
    bad = dict(_snapshot(), symbol='NFO:NIFTY26MAY24050CE', ltp=80.0, bid=79.5, ask=80.5, ltp_only_fallback=False)
    good = dict(_snapshot(), symbol='NFO:NIFTY26MAY24100CE', strike=24100, ltp=150.0, bid=149.0, ask=151.0, ltp_only_fallback=False)
    ranked = selector.select_ranked_candidates(direction_bias='CE', atm_strike=24050, snapshots=[bad, good])
    assert ranked
    assert ranked[0].symbol == 'NFO:NIFTY26MAY24100CE'

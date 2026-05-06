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

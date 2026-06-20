from __future__ import annotations

import logging

import nifty_scalper_bot.strategies.trade_selector as trade_selector_module
from nifty_scalper_bot.strategies.trade_selector import TradeCandidateSelector
from nifty_scalper_bot.utils.log_throttle import DEFAULT_LOG_THROTTLE


def _reset_default_throttle() -> None:
    with DEFAULT_LOG_THROTTLE._lock:
        DEFAULT_LOG_THROTTLE._states.clear()
        DEFAULT_LOG_THROTTLE._change_states.clear()
        DEFAULT_LOG_THROTTLE._last_emit_mono.clear()
        DEFAULT_LOG_THROTTLE._suppressed.clear()


def test_candidate_reject_logging_does_not_raise_and_is_throttled(caplog, monkeypatch) -> None:
    _reset_default_throttle()
    monkeypatch.setenv("CANDIDATE_REJECT_LOG_THROTTLE_SECONDS", "120")
    selector = TradeCandidateSelector()

    with caplog.at_level(logging.INFO):
        selector._log_reject(
            "side_mismatch",
            "NFO:NIFTY1CE",
            throttle_key_parts=("side_mismatch", "NFO:NIFTY1CE", "PE", "CE"),
            side="PE",
            expected_direction="CE",
        )
        selector._log_reject(
            "side_mismatch",
            "NFO:NIFTY1CE",
            throttle_key_parts=("side_mismatch", "NFO:NIFTY1CE", "PE", "CE"),
            side="PE",
            expected_direction="CE",
        )

    records = [
        record
        for record in caplog.records
        if getattr(record, "event", "") == "CANDIDATE_REJECTED"
    ]
    assert len(records) == 1
    assert records[0].message.startswith(
        "CANDIDATE_REJECTED symbol=NFO:NIFTY1CE reason=side_mismatch"
    )


def test_premium_out_of_range_uses_shared_rejection_throttle(caplog, monkeypatch) -> None:
    _reset_default_throttle()
    monkeypatch.setattr(trade_selector_module, "expiry_theta_block", lambda: (False, ""))
    monkeypatch.setattr(trade_selector_module, "midday_pause_block", lambda: (False, ""))
    monkeypatch.setenv("CANDIDATE_REJECT_LOG_THROTTLE_SECONDS", "120")
    selector = TradeCandidateSelector(min_option_premium=40, max_option_premium=650)
    snapshots = [
        {
            "symbol": "NFO:NIFTY1CE",
            "side": "CE",
            "strike": 24000,
            "ltp": 20.0,
            "bid": 19.9,
            "ask": 20.1,
            "tick_age_s": 0.5,
            "real_ticks_last_60s": 5,
        }
    ]

    with caplog.at_level(logging.INFO):
        assert selector.select_ranked_candidates(
            direction_bias="CE", atm_strike=24000, snapshots=snapshots
        ) == []
        assert selector.select_ranked_candidates(
            direction_bias="CE", atm_strike=24000, snapshots=snapshots
        ) == []

    rejection_records = [
        record
        for record in caplog.records
        if getattr(record, "event", "") == "CANDIDATE_REJECTED"
        and getattr(record, "reason", "") == "premium_out_of_range"
    ]
    assert len(rejection_records) == 1


def test_candidate_logging_defaults_are_long_enough_for_production(monkeypatch) -> None:
    monkeypatch.delenv("CANDIDATE_REJECT_LOG_THROTTLE_SECONDS", raising=False)
    monkeypatch.delenv("CANDIDATE_SUMMARY_LOG_THROTTLE_SECONDS", raising=False)
    selector = TradeCandidateSelector()
    assert selector._candidate_reject_log_throttle_seconds == 120.0
    assert selector._candidate_summary_log_throttle_seconds == 300.0

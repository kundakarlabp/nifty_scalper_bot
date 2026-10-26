"""Regression gates for finite candidate configuration and contract snapshots."""

import pytest

from nifty_scalper_bot.risk.cost_model import evaluate_net_reward_risk
from nifty_scalper_bot.strategies.trade_selector import TradeCandidateSelector


def base(**kw):
    s = {
        "symbol": "NFO:NIFTY123CE",
        "side": "CE",
        "strike": 22000,
        "ltp": 120,
        "bid": 119,
        "ask": 121,
        "tick_age_s": 2,
        "real_ticks_last_60s": 3,
        "atr_option": 7,
    }
    s.update(kw)
    return s


def test_stale_rejected():
    r = TradeCandidateSelector(quality_mode="strict").select_ranked_candidates(
        direction_bias="CE", atm_strike=22000, snapshots=[base(tick_age_s=11)]
    )
    assert not r


def test_wide_spread_rejected():
    r = TradeCandidateSelector().select_ranked_candidates(
        direction_bias="CE", atm_strike=22000, snapshots=[base(bid=100, ask=130)]
    )
    assert not r


def test_valid_selected_and_rr():
    r = TradeCandidateSelector().select_ranked_candidates(
        direction_bias="CE", atm_strike=22000, snapshots=[base()]
    )
    assert r and r[0].rr and r[0].rr >= 1.5
    candidate = r[0]
    economics = evaluate_net_reward_risk(
        entry_price=candidate.entry_price,
        stop_price=candidate.stop_loss,
        target_price=candidate.target,
        quantity=65,
        half_spread=1.0,
    )
    assert economics.allowed is True


def test_far_otm_rejected():
    r = TradeCandidateSelector().select_ranked_candidates(
        direction_bias="CE", atm_strike=22000, snapshots=[base(strike=22400)]
    )
    assert not r


def test_tighter_spread_ranks_ahead_when_economics_match():
    selector = TradeCandidateSelector(max_option_spread_pct=1.0)
    ranked = selector.select_ranked_candidates(
        direction_bias="CE",
        atm_strike=22000,
        snapshots=[
            base(symbol="NFO:NIFTYWIDECE", bid=119.64, ask=120.36),
            base(symbol="NFO:NIFTYTIGHTCE", bid=119.88, ask=120.12),
        ],
    )
    assert len(ranked) == 2
    assert ranked[0].symbol == "NFO:NIFTYTIGHTCE"
    assert ranked[0].spread_pct < ranked[1].spread_pct


def test_data_quality_uses_canonical_millisecond_age_over_stale_legacy_seconds():
    selector = TradeCandidateSelector()
    quality = selector.evaluate_data_quality(base(tick_age_s=99, tick_age_ms=250))
    assert quality.allowed is True
    assert quality.reasons == ["data_quality_ok"]


def test_ranked_candidate_reuses_resolved_readiness():
    selector = TradeCandidateSelector()
    ranked = selector.select_ranked_candidates(
        direction_bias="CE",
        atm_strike=22000,
        snapshots=[base(tick_age_s=99, tick_age_ms=250)],
    )
    assert ranked
    assert ranked[0].tick_age_s == 0.25


def test_data_quality_respects_effective_candidate_limits():
    selector = TradeCandidateSelector(max_tick_age_s=1.0, require_real_ticks_last_60s=3)
    quality = selector.evaluate_data_quality(
        base(tick_age_ms=900), max_age=1.0, min_ticks=3, max_spread=5.0
    )
    assert quality.allowed is True
    assert quality.reasons == ["data_quality_ok"]


def test_execution_premium_floor_cannot_be_bypassed_by_dynamic_filter(monkeypatch):
    monkeypatch.setenv("EXECUTION_MIN_OPTION_PREMIUM", "40")
    monkeypatch.setenv("MIN_OPTION_PREMIUM_DYNAMIC", "true")
    monkeypatch.setenv("MIN_OPTION_PREMIUM_DYNAMIC_FLOOR", "25")
    snapshot = base(ltp=30.0, bid=29.9, ask=30.1)
    ranked = TradeCandidateSelector().select_ranked_candidates(
        direction_bias="CE",
        atm_strike=22000,
        snapshots=[snapshot],
    )
    assert ranked == []


def test_crossed_option_book_is_never_an_execution_candidate(monkeypatch):
    import nifty_scalper_bot.strategies.trade_selector as selector_module

    monkeypatch.setattr(selector_module, "expiry_theta_block", lambda: (False, ""))
    monkeypatch.setattr(selector_module, "midday_pause_block", lambda: (False, ""))
    selector = TradeCandidateSelector()
    crossed = base(bid=125, ask=115)

    ranked = selector.select_ranked_candidates(
        direction_bias="CE", atm_strike=22000, snapshots=[crossed]
    )
    assert ranked == []
    quality = selector.evaluate_data_quality(crossed)
    assert quality.allowed is False
    assert "missing_bid_ask" in quality.reasons


def test_canonical_depth_quote_remains_eligible(monkeypatch):
    import nifty_scalper_bot.strategies.trade_selector as selector_module

    monkeypatch.setattr(selector_module, "expiry_theta_block", lambda: (False, ""))
    monkeypatch.setattr(selector_module, "midday_pause_block", lambda: (False, ""))
    selector = TradeCandidateSelector()
    depth_quote = base(
        bid=None,
        ask=None,
        depth={
            "buy": [{"price": 119.0}],
            "sell": [{"price": 121.0}],
        },
    )

    candidates = selector.select_ranked_candidates(
        direction_bias="CE", atm_strike=22000, snapshots=[depth_quote]
    )
    assert len(candidates) == 1
    assert candidates[0].entry_price == 121.0
    assert selector.evaluate_data_quality(depth_quote).allowed is True


def test_unusable_timestamp_quality_cannot_prove_candidate_bid_ask(monkeypatch):
    import nifty_scalper_bot.strategies.trade_selector as selector_module

    monkeypatch.setattr(selector_module, "expiry_theta_block", lambda: (False, ""))
    monkeypatch.setattr(selector_module, "midday_pause_block", lambda: (False, ""))
    selector = TradeCandidateSelector()
    snapshot = base(timestamp_quality="synthetic")

    ranked = selector.select_ranked_candidates(
        direction_bias="CE", atm_strike=22000, snapshots=[snapshot]
    )
    assert ranked == []
    assert "missing_bid_ask" in selector.evaluate_data_quality(snapshot).reasons


def test_nonfinite_option_ltp_is_not_executable(monkeypatch):
    import nifty_scalper_bot.strategies.trade_selector as selector_module

    monkeypatch.setattr(selector_module, "expiry_theta_block", lambda: (False, ""))
    monkeypatch.setattr(selector_module, "midday_pause_block", lambda: (False, ""))
    selector = TradeCandidateSelector()
    for value in (float("nan"), float("inf"), float("-inf")):
        ranked = selector.select_ranked_candidates(
            direction_bias="CE", atm_strike=22000, snapshots=[base(ltp=value)]
        )
        assert ranked == []


def test_candidate_live_mode_is_owned_by_entry_policy(monkeypatch):
    from nifty_scalper_bot.config.entry_policy import resolve_entry_policy

    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "false")
    monkeypatch.setenv("PAPER_MODE", "false")
    monkeypatch.setenv("PAPER__ENABLED", "false")
    monkeypatch.setenv("SHADOW_MODE", "false")

    for live_flag in ("1", "false", "live"):
        monkeypatch.setenv("ENABLE_LIVE", live_flag)
        policy = resolve_entry_policy()
        selector = TradeCandidateSelector()
        max_spread, _max_age, _min_ticks, is_live = selector._effective_limits()
        assert is_live is policy.is_live
        if is_live:
            assert max_spread <= policy.execution_max_spread_pct


def test_candidate_uses_canonical_quote_spread_value(monkeypatch):
    import nifty_scalper_bot.strategies.trade_selector as selector_module
    from nifty_scalper_bot.execution.readiness import resolve_quote_bid_ask_spread

    monkeypatch.setattr(selector_module, "expiry_theta_block", lambda: (False, ""))
    monkeypatch.setattr(selector_module, "midday_pause_block", lambda: (False, ""))
    snapshot = base(bid=119.9, ask=120.1)
    expected = resolve_quote_bid_ask_spread(snapshot)[2]

    ranked = TradeCandidateSelector().select_ranked_candidates(
        direction_bias="CE", atm_strike=22000, snapshots=[snapshot]
    )
    assert len(ranked) == 1
    assert ranked[0].spread_pct == expected


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("min_option_premium", float("nan")),
        ("min_option_premium", float("inf")),
        ("max_option_premium", float("nan")),
        ("max_option_premium", float("-inf")),
        ("max_tick_age_s", float("nan")),
        ("max_tick_age_s", float("inf")),
        ("max_tick_age_s", 0.0),
        ("max_option_spread_pct", float("nan")),
        ("max_option_spread_pct", float("inf")),
        ("max_option_spread_pct", -1.0),
    ],
)
def test_nonfinite_or_invalid_candidate_limits_are_rejected(field, invalid):
    with pytest.raises(ValueError):
        TradeCandidateSelector(**{field: invalid})


@pytest.mark.parametrize(
    "strike", [None, "bad", float("nan"), float("inf"), 22000.5, 0]
)
def test_bad_candidate_strike_is_rejected_not_raised(monkeypatch, strike):
    import nifty_scalper_bot.strategies.trade_selector as selector_module

    monkeypatch.setattr(selector_module, "expiry_theta_block", lambda: (False, ""))
    monkeypatch.setattr(selector_module, "midday_pause_block", lambda: (False, ""))
    selector = TradeCandidateSelector()
    assert selector.select_ranked_candidates(
        direction_bias="CE", atm_strike=22000, snapshots=[base(strike=strike)]
    ) == []
    assert selector._last_rejects["invalid_strike"] == 1


@pytest.mark.parametrize(
    ("atm", "rr"),
    [
        (float("nan"), 2.0),
        (float("inf"), 2.0),
        (22000, float("nan")),
        (22000, float("inf")),
    ],
)
def test_invalid_selection_context_cannot_admit_candidate(monkeypatch, atm, rr):
    import nifty_scalper_bot.strategies.trade_selector as selector_module

    monkeypatch.setattr(selector_module, "expiry_theta_block", lambda: (False, ""))
    monkeypatch.setattr(selector_module, "midday_pause_block", lambda: (False, ""))
    selector = TradeCandidateSelector()
    assert selector.select_ranked_candidates(
        direction_bias="CE", atm_strike=atm, snapshots=[base()], gross_rr=rr
    ) == []
    assert selector._last_rejects["invalid_selection_context"] == 1


def test_nonfinite_live_age_env_cannot_relax_bound(monkeypatch):
    import nifty_scalper_bot.strategies.trade_selector as selector_module

    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("LIVE_CANDIDATE_MAX_TICK_AGE_S", "nan")
    selector = TradeCandidateSelector()
    _spread, age, _ticks, is_live = selector._effective_limits()
    assert is_live is True
    assert age <= 2.5
    assert selector_module._float_env("LIVE_CANDIDATE_MAX_TICK_AGE_S", 2.5) == 2.5

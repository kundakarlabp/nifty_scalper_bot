from nifty_scalper_bot.core.strategy_manager import StrategyManager, StrategyVote, _enrich_option_candidate_metadata
from nifty_scalper_bot.strategies.signal_generator import Signal


def test_candidate_metadata_derives_strike_distance_from_same_side_selected():
    enriched = _enrich_option_candidate_metadata(
        "NFO:NIFTY26MAY24050CE",
        {"selected_ce": "NFO:NIFTY26MAY24000CE", "selected_pe": "NFO:NIFTY26MAY24000PE"},
    )

    assert enriched["selected_ce"] == "NFO:NIFTY26MAY24000CE"
    assert enriched["selected_pe"] == "NFO:NIFTY26MAY24000PE"
    assert enriched["is_selected_option"] is False
    assert enriched["strike_distance_from_atm"] == 50.0


def _signal_vote(distance: float):
    metadata = {
        "strategy": "OrderFlow",
        "role": "trigger",
        "strategy_score": 8.0,
        "raw_setup_score": 8.0,
        "quote_depth_valid": True,
        "spread_pct": 0.5,
        "selected_ce": "NFO:NIFTY26MAY24000CE",
        "selected_pe": "NFO:NIFTY26MAY24000PE",
        "is_selected_option": False,
        "strike_distance_from_atm": distance,
        "trigger_conditions_met": True,
    }
    signal = Signal(
        action="BUY",
        symbol="NFO:NIFTY26MAY24050CE",
        quantity=1,
        confidence=0.85,
        reason="orderflow",
        stop_loss=None,
        take_profit=None,
        metadata=dict(metadata),
    )
    vote = StrategyVote(
        strategy="OrderFlow",
        side="CE",
        score=8.0,
        confidence=0.85,
        reasons=["depth"],
        metadata=dict(metadata),
    )
    return signal, vote


def test_high_score_candidate_switch_only_within_configured_distance(monkeypatch):
    monkeypatch.setenv("STRATEGY_ALLOW_CANDIDATE_SWITCH_ON_HIGH_SCORE", "true")
    monkeypatch.setenv("CANDIDATE_SWITCH_MAX_DISTANCE_POINTS", "75")
    monkeypatch.setenv("LIVE_MAX_SPREAD_PCT", "0.75")
    manager = StrategyManager()

    near = manager._combine_strategy_votes(
        symbol="NFO:NIFTY26MAY24050CE",
        signals=[_signal_vote(50.0)],
        indicators={"quote_depth_valid": True, "spread_pct": 0.5},
    )
    far = manager._combine_strategy_votes(
        symbol="NFO:NIFTY26MAY24150CE",
        signals=[_signal_vote(150.0)],
        indicators={"quote_depth_valid": True, "spread_pct": 0.5},
    )

    assert near is not None
    assert near.metadata["candidate_switch_requested"] is True
    assert near.metadata["candidate_switch_reason"] == "high_score_nearby_option_candidate"
    assert far is None

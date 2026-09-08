from types import SimpleNamespace

from nifty_scalper_bot.core.strategy_vote_policy import (
    independent_same_side_confirmation,
    partition_votes,
    setup_gate_decision,
    vote_role,
)


def _vote(strategy="ORBPro", side="CE", score=6.0, **metadata):
    return SimpleNamespace(strategy=strategy, side=side, score=score, metadata=metadata)


def _signal(action="BUY"):
    return SimpleNamespace(action=action)


def test_orderflow_is_context_only_even_if_legacy_metadata_claims_trigger():
    vote = _vote(strategy="OrderFlow", role="trigger", can_trigger=True)
    assert vote_role(vote) == "context"


def test_setup_contract_fails_closed_below_own_minimum():
    decision = setup_gate_decision(
        _vote(raw_setup_score=4.9, setup_min=5.5, setup_pass=True)
    )
    assert decision.passed is False
    assert decision.reason == "setup_below_minimum"


def test_partition_never_promotes_context_to_trigger():
    trigger = (_signal(), _vote(strategy="ORBPro", role="trigger", raw_setup_score=6.0, setup_min=5.5, setup_pass=True))
    context = (_signal(), _vote(strategy="OrderFlow", role="trigger", score=9.0))
    triggers, contexts, rejected = partition_votes([trigger, context])
    assert triggers == [trigger]
    assert contexts == [context]
    assert rejected == []


def test_failed_setup_cannot_be_rescued_by_context():
    weak = (_signal(), _vote(strategy="VWAPPro", role="trigger", raw_setup_score=4.0, setup_min=5.0, setup_pass=True))
    context = (_signal(), _vote(strategy="OrderFlow", role="context", score=10.0))
    triggers, contexts, rejected = partition_votes([weak, context])
    assert triggers == []
    assert contexts == [context]
    assert rejected[0]["strategy"] == "VWAPPro"


def test_close_signal_bypasses_entry_setup_gate():
    close = (_signal("CLOSE_LONG"), _vote(strategy="VWAPPro", role="trigger", raw_setup_score=1.0, setup_min=5.0, setup_pass=False))
    triggers, contexts, rejected = partition_votes([close])
    assert triggers == [close]
    assert contexts == []
    assert rejected == []


def test_confirmation_requires_distinct_same_side_trigger_strategy():
    signals = [
        (_signal(), _vote(strategy="ORBPro", side="CE", role="trigger")),
        (_signal(), _vote(strategy="VWAPPro", side="CE", score=5.8, role="trigger")),
        (_signal(), _vote(strategy="OrderFlow", side="CE", score=9.0, role="trigger")),
    ]
    confirmed, strategies = independent_same_side_confirmation(signals)
    assert confirmed is True
    assert strategies == ["VWAPPro"]


def test_opposite_side_or_context_vote_is_not_confirmation():
    signals = [
        (_signal(), _vote(strategy="ORBPro", side="CE", role="trigger")),
        (_signal(), _vote(strategy="VWAPPro", side="PE", role="trigger")),
        (_signal(), _vote(strategy="OrderFlow", side="CE", score=9.0, role="trigger")),
    ]
    assert independent_same_side_confirmation(signals) == (False, [])


def test_failed_setup_trigger_cannot_count_as_independent_confirmation():
    signals = [
        (_signal(), _vote(strategy="ORBPro", side="CE", score=8.0, role="trigger", raw_setup_score=8.0, setup_min=5.5, setup_pass=True)),
        (_signal(), _vote(strategy="VWAPPro", side="CE", score=9.0, role="trigger", raw_setup_score=4.0, setup_min=5.8, setup_pass=False)),
    ]
    assert independent_same_side_confirmation(signals) == (False, [])


def test_close_vote_cannot_count_as_independent_entry_confirmation():
    signals = [
        (_signal(), _vote(strategy="ORBPro", side="CE", score=8.0, role="trigger")),
        (_signal("CLOSE_LONG"), _vote(strategy="VWAPPro", side="CE", score=9.0, role="trigger")),
    ]
    assert independent_same_side_confirmation(signals) == (False, [])

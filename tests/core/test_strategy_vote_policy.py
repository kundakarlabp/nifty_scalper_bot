from types import SimpleNamespace

from nifty_scalper_bot.core.strategy_vote_policy import (
    independent_same_side_confirmation,
    is_permanent_context_only,
    partition_votes,
    setup_gate_decision,
    vote_role,
)


def _evidence(strategy="ORBPro", side="CE", **metadata):
    return SimpleNamespace(
        strategy=strategy,
        side=side,
        reasons=list(metadata.pop("reasons", [])),
        metadata=metadata,
    )


def _signal(action="BUY"):
    return SimpleNamespace(action=action)


def test_setup_gate_requires_explicit_structural_pass() -> None:
    passed = setup_gate_decision(
        _evidence(role="trigger", setup_pass=True, required_data_present=True)
    )
    failed = setup_gate_decision(
        _evidence(
            role="trigger",
            setup_pass=False,
            trigger_block_reason="not_confirmed",
        )
    )
    assert passed.passed is True
    assert failed.passed is False
    assert failed.reason == "not_confirmed"


def test_side_conflict_fails_closed() -> None:
    decision = setup_gate_decision(
        _evidence(role="trigger", setup_pass=True, side_conflict=True)
    )
    assert decision.passed is False
    assert decision.reason == "strategy_contract_side_conflict"


def test_context_only_taxonomy_cannot_be_promoted_by_metadata() -> None:
    context = _evidence(strategy="OrderFlow", role="trigger", setup_pass=True)
    assert is_permanent_context_only(context) is True
    assert vote_role(context) == "context"


def test_partition_separates_valid_triggers_context_and_rejections() -> None:
    valid = (_signal(), _evidence(strategy="ORBPro", role="trigger", setup_pass=True))
    context = (_signal(), _evidence(strategy="OrderFlow", role="context"))
    rejected = (
        _signal(),
        _evidence(
            strategy="VWAPPro",
            role="trigger",
            setup_pass=False,
            trigger_block_reason="event_unconfirmed",
        ),
    )
    triggers, contexts, rejected_setups = partition_votes([valid, context, rejected])
    assert triggers == [valid]
    assert contexts == [context]
    assert rejected_setups == [{"strategy": "VWAPPro", "reason": "event_unconfirmed"}]


def test_close_signal_bypasses_entry_setup_gate() -> None:
    close = (
        _signal("CLOSE_LONG"),
        _evidence(strategy="VWAPPro", role="trigger", setup_pass=False),
    )
    triggers, contexts, rejected = partition_votes([close])
    assert triggers == [close]
    assert contexts == []
    assert rejected == []


def test_independent_confirmation_requires_distinct_signal_families() -> None:
    smc = (
        _signal(),
        _evidence(strategy="SMC", side="CE", role="trigger", setup_pass=True),
    )
    orb = (
        _signal(),
        _evidence(strategy="ORBPro", side="CE", role="trigger", setup_pass=True),
    )
    ok, names = independent_same_side_confirmation([smc, orb])
    assert ok is True
    assert names == ["ORBPro"]


def test_same_family_or_opposite_side_does_not_confirm() -> None:
    smc = (
        _signal(),
        _evidence(strategy="SMC", side="CE", role="trigger", setup_pass=True),
    )
    vwap = (
        _signal(),
        _evidence(strategy="VWAPPro", side="CE", role="trigger", setup_pass=True),
    )
    pe = (
        _signal(),
        _evidence(strategy="ORBPro", side="PE", role="trigger", setup_pass=True),
    )
    same_family_ok, _ = independent_same_side_confirmation([smc, vwap])
    opposite_ok, _ = independent_same_side_confirmation([smc, pe])
    assert same_family_ok is False
    assert opposite_ok is False

from types import SimpleNamespace

from nifty_scalper_bot.config.strategy_taxonomy import canonical_strategy_role
from nifty_scalper_bot.core.strategy_vote_policy import (
    independent_same_side_confirmation,
    is_permanent_context_only,
    partition_votes,
    range_legacy_context_mismatch_only,
    setup_gate_decision,
    verified_range_vwap_final_order_risk,
    verified_range_vwap_reversal,
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


def test_live_range_role_requires_explicit_real_broker_pilot(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("ENABLE_RANGE_VWAP_LIVE", "true")
    monkeypatch.setenv("PAPER__ENABLED", "true")
    assert canonical_strategy_role("range_vwap") == "context"
    monkeypatch.setenv("PAPER__ENABLED", "false")
    monkeypatch.setenv("PAPER_MODE", "false")
    monkeypatch.setenv("SHADOW_MODE", "false")
    assert canonical_strategy_role("RangeVWAPReclaim") == "trigger"
    monkeypatch.setenv("ENABLE_RANGE_VWAP_LIVE", "false")
    assert canonical_strategy_role("range_vwap") == "context"


def test_live_range_reversal_requires_proven_structure_flow_and_risk(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("ENABLE_RANGE_VWAP_LIVE", "true")
    monkeypatch.setenv("PAPER__ENABLED", "false")
    monkeypatch.setenv("PAPER_MODE", "false")
    monkeypatch.setenv("SHADOW_MODE", "false")
    metadata = {
        "role": "trigger",
        "setup_pass": True,
        "range_option_flow_verified": True,
        "range_structure": {
            "qualified": True,
            "side": "CE",
            "reversal_direction_source": "two_completed_futures_candles",
        },
        "option_cost_risk": {
            "qualified": True,
            "model_only": True,
            "worst_stop_loss_including_costs": 700.0,
            "net_reward_risk": 1.6,
        },
    }
    assert verified_range_vwap_reversal(
        metadata, strategy="RangeVWAPReclaim", side="CE"
    )
    assert not verified_range_vwap_reversal(metadata, strategy="SMC", side="CE")
    assert not verified_range_vwap_reversal(
        metadata, strategy="RangeVWAPReclaim", side="PE"
    )
    assert not verified_range_vwap_reversal(
        {**metadata, "range_option_flow_verified": False},
        strategy="RangeVWAPReclaim",
        side="CE",
    )
    assert not verified_range_vwap_reversal(
        {
            **metadata,
            "option_cost_risk": {
                **metadata["option_cost_risk"],
                "worst_stop_loss_including_costs": 774.0,
            },
        },
        strategy="RangeVWAPReclaim",
        side="CE",
    )
    assert not verified_range_vwap_reversal(
        {
            **metadata,
            "option_cost_risk": {
                **metadata["option_cost_risk"],
                "net_reward_risk": 1.49,
            },
        },
        strategy="RangeVWAPReclaim",
        side="CE",
    )
    triggers, contexts, rejected = partition_votes(
        [
            (
                _signal(),
                _evidence(
                    strategy="RangeVWAPReclaim",
                    side="CE",
                    **metadata,
                ),
            )
        ]
    )
    assert len(triggers) == 1
    assert not contexts
    assert not rejected


def test_only_old_trend_bias_conflict_is_exempt_for_verified_range() -> None:
    flow = {
        "context_alignment_block_reason": "underlying_direction_mismatch",
        "candidate_symbol": "NFO:NIFTY26NOV25250CE",
        "quote_update_version": 123,
        "quote_readiness_allowed": True,
        "ofi_ready": True,
        "ofi_3s_complete": True,
        "range_research_holdoff_active": False,
        "microstructure_conflicts_side": False,
        "strong_depth_conflicts_side": False,
        "ofi_conflicts_side": False,
        "ofi_3s_span_seconds": 2.0,
        "ofi_update_count_3s": 4,
        "ofi_3s_normalized": 0.17,
        "ofi_1s_normalized": 0.06,
    }

    def allowed(**overrides):
        return range_legacy_context_mismatch_only(
            _evidence(strategy="OrderFlow", **{**flow, **overrides}),
            expected_symbol="NFO:NIFTY26NOV25250CE",
            expected_quote_version=123,
        )

    assert allowed()
    assert not allowed(microstructure_conflicts_side=True)
    assert not allowed(ofi_conflicts_side=True)
    assert not allowed(quote_update_version=122)
    assert not allowed(ofi_3s_normalized=-0.2)
    assert not allowed(ofi_3s_complete=False)
    assert not allowed(context_alignment_block_reason="adverse_microstructure")
    assert not allowed(range_research_holdoff_active=True)


def test_range_live_final_repricing_respects_actual_broker_lot_and_cash(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("ENABLE_RANGE_VWAP_LIVE", "true")
    monkeypatch.setenv("PAPER__ENABLED", "false")
    monkeypatch.setenv("PAPER_MODE", "false")
    monkeypatch.setenv("SHADOW_MODE", "false")
    base = {
        "role": "trigger",
        "setup_pass": True,
        "range_option_flow_verified": True,
        "bracket_anchor_mode": "absolute_level",
        "range_structure": {
            "qualified": True,
            "side": "CE",
            "reversal_direction_source": "two_completed_futures_candles",
        },
        "option_cost_risk": {
            "qualified": True,
            "model_only": True,
            "worst_stop_loss_including_costs": 650.0,
            "net_reward_risk": 1.6,
            "observed_contract_lot_size": 65,
            "slippage_each_side_bps": 25.0,
            "stop_price_model": 91.0,
            "modeled_stressed_stop_fill_price": 90.5,
            "modeled_extra_exit_spread_points": 0.2,
            "vwap_limited_target_price_model": 120.0,
        },
    }
    args = {
        "strategy": "RangeVWAPReclaim",
        "symbol": "NFO:NIFTY26NOV25250CE",
        "final_symbol": "NFO:NIFTY26NOV25250CE",
        "broker_quantity": 65,
        "broker_lot_size": 65,
        "entry_price": 100.0,
        "ask_price": 100.0,
        "stop_price": 91.0,
        "target_price": 120.0,
    }
    assert verified_range_vwap_final_order_risk(base, **args)
    assert not verified_range_vwap_final_order_risk(
        {**base, "range_counter_broad_trend": True}, **args
    )
    reduced_payoff = {
        **base,
        "option_cost_risk": {
            **base["option_cost_risk"],
            "net_reward_risk": 1.8,
            "vwap_limited_target_price_model": 118.0,
        },
    }
    reduced_args = {**args, "target_price": 118.0}
    assert verified_range_vwap_final_order_risk(reduced_payoff, **reduced_args)
    assert not verified_range_vwap_final_order_risk(
        {**reduced_payoff, "range_counter_broad_trend": True}, **reduced_args
    )
    assert not verified_range_vwap_final_order_risk(
        base, **{**args, "broker_lot_size": 75}
    )
    assert not verified_range_vwap_final_order_risk(
        base, **{**args, "broker_quantity": 130}
    )
    assert not verified_range_vwap_final_order_risk(
        base, **{**args, "stop_price": 85.0}
    )
    assert not verified_range_vwap_final_order_risk(
        base, **{**args, "final_symbol": "NFO:NIFTY26NOV25300CE"}
    )
    assert not verified_range_vwap_final_order_risk(
        base, **{**args, "target_price": 122.0}
    )
    assert not verified_range_vwap_final_order_risk(
        {**base, "bracket_anchor_mode": "distance"}, **args
    )
    assert not verified_range_vwap_final_order_risk(
        base, **{**args, "entry_price": 106.0}
    )
    # A modeled liquidation fill must be below the stop trigger; missing or
    # optimistic fill proof cannot silently pass the final broker risk gate.
    for modeled_fill in (91.0, 95.0, -1.0):
        corrupted = {
            **base,
            "option_cost_risk": {
                **base["option_cost_risk"],
                "modeled_stressed_stop_fill_price": modeled_fill,
            },
        }
        assert not verified_range_vwap_final_order_risk(corrupted, **args)
    unproven = {**base, "option_cost_risk": {**base["option_cost_risk"]}}
    del unproven["option_cost_risk"]["modeled_stressed_stop_fill_price"]
    assert not verified_range_vwap_final_order_risk(unproven, **args)


def test_regime_dispatch_preserves_structural_strategy_ownership() -> None:
    from nifty_scalper_bot.config.strategy_taxonomy import strategy_regime_eligible

    assert strategy_regime_eligible("RangeVWAPReclaim", "RANGE")
    assert not strategy_regime_eligible("RangeVWAPReclaim", "TREND")
    assert strategy_regime_eligible("VWAPPro", "TREND")
    assert not strategy_regime_eligible("VWAPPro", "RANGE")
    assert strategy_regime_eligible("SMC", "RANGE")
    assert not strategy_regime_eligible("ORBPro", "RANGE")
    assert not strategy_regime_eligible("SMC", "UNKNOWN")
    assert strategy_regime_eligible("OrderFlow", "RANGE")


def test_orb_range_transition_remains_shadow_research_only(monkeypatch) -> None:
    from nifty_scalper_bot.config.strategy_taxonomy import strategy_regime_eligible

    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("ORB_RESEARCH_ALLOW_RANGE_TRANSITION", "true")
    assert strategy_regime_eligible("ORBPro", "RANGE")
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    assert not strategy_regime_eligible("ORBPro", "RANGE")



def test_nested_range_pilot_requires_separate_opt_in_and_local_proof(monkeypatch) -> None:
    from nifty_scalper_bot.config.strategy_taxonomy import strategy_regime_eligible

    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("ENABLE_RANGE_VWAP_LIVE", "true")
    monkeypatch.setenv("PAPER__ENABLED", "false")
    monkeypatch.setenv("PAPER_MODE", "false")
    monkeypatch.setenv("SHADOW_MODE", "false")
    monkeypatch.setenv("ENABLE_RANGE_VWAP_LOCAL_TREND_LIVE", "false")
    metadata = {
        "role": "trigger",
        "setup_pass": True,
        "range_option_flow_verified": True,
        "range_local_trend_pilot": True,
        "range_spot_broad_trend_side": "PE",
        "range_structure": {
            "qualified": True,
            "side": "CE",
            "reversal_direction_source": "two_completed_futures_candles",
            "local_range_confirmed": True,
            "local_reclaim_side": "CE",
            "broader_60bar_trend_side": "PE",
        },
        "option_cost_risk": {
            "qualified": True,
            "model_only": True,
            "worst_stop_loss_including_costs": 700.0,
            "net_reward_risk": 1.8,
        },
    }
    assert not strategy_regime_eligible("RangeVWAPReclaim", "TREND")
    assert not verified_range_vwap_reversal(
        metadata, strategy="RangeVWAPReclaim", side="CE"
    )
    monkeypatch.setenv("ENABLE_RANGE_VWAP_LOCAL_TREND_LIVE", "true")
    assert strategy_regime_eligible("RangeVWAPReclaim", "TREND")
    assert verified_range_vwap_reversal(
        metadata, strategy="RangeVWAPReclaim", side="CE"
    )
    assert not verified_range_vwap_reversal(
        {**metadata, "range_spot_broad_trend_side": "CE"},
        strategy="RangeVWAPReclaim",
        side="CE",
    )
    assert not verified_range_vwap_reversal(
        {
            **metadata,
            "range_structure": {
                **metadata["range_structure"],
                "local_range_confirmed": False,
            },
        },
        strategy="RangeVWAPReclaim",
        side="CE",
    )

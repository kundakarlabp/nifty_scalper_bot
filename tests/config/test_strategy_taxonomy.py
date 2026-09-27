from nifty_scalper_bot.config.strategy_taxonomy import (
    canonical_signal_family,
    canonical_strategy_role,
    is_context_only_strategy,
    normalize_strategy_name,
)


def test_aliases_share_one_structural_identity() -> None:
    assert normalize_strategy_name("OrderFlow") == "order_flow"
    assert normalize_strategy_name("order_flow") == "order_flow"
    assert normalize_strategy_name("CPR") == "cpr_breakout"
    assert normalize_strategy_name("RSIDivergence") == "rsi_divergence"


def test_directional_runtime_roles_are_canonical() -> None:
    for name in ("SMC", "VWAPPro", "ORBPro"):
        assert canonical_strategy_role(name) == "trigger"
        assert canonical_signal_family(name) == "directional_trigger"

    for name in ("OrderFlow", "OIMaxPain", "BBSqueeze", "CPRBreakout", "RSIDivergence"):
        assert is_context_only_strategy(name)
        assert canonical_signal_family(name) == "directional_context"


def test_unknown_strategy_defaults_fail_compatible() -> None:
    assert canonical_strategy_role("LegacyStrategy") == "trigger"

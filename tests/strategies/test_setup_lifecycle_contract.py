# Canonical lifecycle is observability-only; native owners retain all trading decisions.
from __future__ import annotations

from nifty_scalper_bot.strategies.setup_lifecycle import (
    SetupLifecycleRegistry,
    SetupStage,
)


def test_setup_lifecycle_is_monotonic_and_terminal() -> None:
    registry = SetupLifecycleRegistry(limit=128)
    kwargs = {
        "strategy": "VWAPPro",
        "setup_id": "vwap:CE:anchor",
        "symbol": "NFO:NIFTY26SEP22650CE",
        "side": "CE",
    }
    registry.transition(SetupStage.ARMED, **kwargs)
    registry.transition(SetupStage.CONFIRMING, reason="waiting", **kwargs)
    registry.transition(SetupStage.TRIGGER_QUALIFIED, **kwargs)
    registry.transition(SetupStage.MANAGER_QUALIFIED, **kwargs)
    registry.transition(SetupStage.RUNNER_APPROVED, **kwargs)
    registry.transition(SetupStage.TRADE_PLAN, **kwargs)
    accepted = registry.transition(SetupStage.BROKER_ACCEPTED, reason="OID-1", **kwargs)
    after_terminal = registry.transition(SetupStage.CONFIRMING, reason="late", **kwargs)

    assert accepted is not None
    assert after_terminal is accepted
    assert accepted.stage == SetupStage.BROKER_ACCEPTED.value
    snapshot = registry.snapshot()
    assert snapshot["active_count"] == 0
    assert snapshot["terminal_counts"][SetupStage.BROKER_ACCEPTED.value] == 1


def test_setup_lifecycle_rejects_nonterminal_regression() -> None:
    registry = SetupLifecycleRegistry(limit=128)
    kwargs = {"strategy": "SMC", "setup_id": "smcv2:1", "side": "PE"}
    qualified = registry.transition(SetupStage.MANAGER_QUALIFIED, **kwargs)
    regressed = registry.transition(SetupStage.CONFIRMING, **kwargs)

    assert qualified is not None
    assert regressed is qualified
    assert regressed.stage == SetupStage.MANAGER_QUALIFIED.value


def test_setup_lifecycle_requires_structural_identity() -> None:
    registry = SetupLifecycleRegistry(limit=128)
    assert (
        registry.transition(
            SetupStage.ARMED,
            strategy="ORBPro",
            setup_id="",
            symbol="NFO:NIFTY26SEP22650CE",
            side="CE",
        )
        is None
    )


def test_setup_lifecycle_terminal_rejection_is_counted() -> None:
    registry = SetupLifecycleRegistry(limit=128)
    registry.transition(
        SetupStage.CONTRACT_REJECTED,
        strategy="VWAPPro",
        setup_id="vwap:PE:anchor",
        side="PE",
        reason="setup_contract_not_passed",
    )
    snapshot = registry.snapshot()
    assert snapshot["terminal_counts"][SetupStage.CONTRACT_REJECTED.value] == 1
    assert snapshot["transition_counts"][SetupStage.CONTRACT_REJECTED.value] == 1


def test_native_strategy_sources_cover_direct_and_terminal_lifecycle_paths() -> None:
    from pathlib import Path

    root = Path("src/nifty_scalper_bot/strategies/elite_strategies")
    orb = (root / "orb_pro.py").read_text()
    vwap = (root / "vwap_pro.py").read_text()
    smc = (root / "smc_liquidity.py").read_text()

    # ORB momentum can trigger without a retest; it must still start at ARMED.
    momentum_idx = orb.index("if momentum_confirmed:")
    momentum = orb[momentum_idx - 700 : momentum_idx + 900]
    assert "SetupStage.ARMED" in momentum
    assert "SetupStage.CONFIRMING" in momentum
    assert "SetupStage.CONTRACT_REJECTED" in momentum

    # A confirmed VWAP event must arm before confirmation and every failed
    # structural prerequisite must terminate the lifecycle explicitly.
    setup_idx = vwap.index("setup_lifecycle_id =")
    unconfirmed_idx = vwap.index("if not event_confirmed:", setup_idx)
    assert "SetupStage.ARMED" in vwap[setup_idx:unconfirmed_idx]
    rejected_idx = vwap.index("if structural_failures:", unconfirmed_idx)
    assert "SetupStage.CONTRACT_REJECTED" in vwap[rejected_idx : rejected_idx + 700]

    # SMC structural sweeps remain the native owner of setup formation.
    assert "SetupStage.ARMED" in smc
    assert "SetupStage.CONFIRMING" in smc

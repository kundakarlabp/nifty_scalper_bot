from __future__ import annotations

from types import SimpleNamespace

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
        SetupStage.QUALITY_REJECTED,
        strategy="VWAPPro",
        setup_id="vwap:PE:anchor",
        side="PE",
        reason="alpha_below_threshold",
    )
    snapshot = registry.snapshot()
    assert snapshot["terminal_counts"][SetupStage.QUALITY_REJECTED.value] == 1
    assert snapshot["transition_counts"][SetupStage.QUALITY_REJECTED.value] == 1
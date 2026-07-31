"""OrderFlow is context only. It cannot generate trade signals.

Trade signal generation belongs exclusively to three setup families -- SMC
liquidity, VWAP Pro and ORB Pro -- graded against min_score ~5.5-5.8.
OrderFlow/OI/IV supply measured confirmation.

31 Jul evidence: EVERY executed trade carried
`SIGNAL_GENERATED ... reason=OrderFlow` while the graded strategies correctly
refused (score 1.50 x46, 3.50 x15, 4.00 x11, 5.00 x9 against min_score
5.80/5.50). OrderFlow ran its own far weaker ladder
(ORDERFLOW_CONTEXT_MIN_SCORE default 4.0), whose LTP fallback awards 2.0
unconditionally, +2.0 for spread <= 12%, +1.5 for direction agreement and +1.0
merely for a tick having a direction -- so an ordinary ticking market reaches
exactly 4.0 and passes.

Two independent code paths granted the trigger role, and BOTH are removed:
  1. order_flow.py  -- ORDERFLOW_ALLOW_LIVE_TRIGGER / ORDERFLOW_ALLOW_TRIGGER_ROLE
     defaulted to 'true', granting a trigger role out of the box;
  2. builder.py     -- ORDERFLOW_ALLOW_TRIGGER_ROLE REMOVED OrderFlow from the
     context list, which promoted it into trigger_names.

The capability is removed, not defaulted off: a role this consequential must
not be re-enablable by environment.
"""

from __future__ import annotations

import inspect

import pytest

from nifty_scalper_bot.strategies.elite_strategies import order_flow
from nifty_scalper_bot.strategies.elite_strategies.builder import (
    _production_strategy_roles,
)

_SETUPS = ["SMCLiquidity", "VWAPPro", "ORBPro"]
_ALL = _SETUPS + ["OrderFlow"]


@pytest.mark.parametrize("mode", ["production", "elite"])
def test_orderflow_is_always_context_never_a_trigger(mode: str) -> None:
    """THE FIX: OrderFlow may not appear among trigger strategies."""
    triggers, context = _production_strategy_roles(_ALL, strategy_mode=mode)
    assert "OrderFlow" not in triggers
    assert "OrderFlow" in context


@pytest.mark.parametrize("mode", ["production", "elite"])
def test_env_cannot_promote_orderflow_to_a_trigger(monkeypatch, mode: str) -> None:
    """The removed switch must not resurrect the role inversion."""
    monkeypatch.setenv("ORDERFLOW_ALLOW_TRIGGER_ROLE", "true")
    monkeypatch.setenv("ORDERFLOW_ALLOW_LIVE_TRIGGER", "true")
    triggers, context = _production_strategy_roles(_ALL, strategy_mode=mode)
    assert "OrderFlow" not in triggers
    assert "OrderFlow" in context


@pytest.mark.parametrize("mode", ["production", "elite"])
def test_three_setup_families_remain_the_trigger_strategies(mode: str) -> None:
    """Signal generation stays with SMC, VWAP Pro and ORB Pro."""
    triggers, _ = _production_strategy_roles(_ALL, strategy_mode=mode)
    for setup in _SETUPS:
        assert setup in triggers


def test_trigger_permission_is_no_longer_environment_controlled() -> None:
    """order_flow.py must not read a trigger-permission env var."""
    src = inspect.getsource(order_flow)
    # The permission must be a constant, not an env lookup.
    assert "allow_orderflow_trigger = False" in src
    assigns = [
        ln.strip() for ln in src.splitlines()
        if ln.strip().startswith("allow_orderflow_trigger =")
    ]
    assert assigns == ["allow_orderflow_trigger = False"], assigns
    assert not any("getenv" in ln for ln in assigns)


def test_context_role_is_reported_as_the_block_reason() -> None:
    """Blocked triggers must read as a role statement, not a toggle."""
    src = inspect.getsource(order_flow)
    assert "'context_only_role'" in src
    assert "'trigger_role_disabled'" not in src


def test_confirmation_scoring_path_is_retained() -> None:
    """OrderFlow still computes and publishes confirmation for the setups."""
    src = inspect.getsource(order_flow)
    assert "ORDERFLOW_CONTEXT_MIN_SCORE" in src
    assert "context_min_score" in src


def test_three_setup_modules_exist_and_are_distinct() -> None:
    from nifty_scalper_bot.strategies.elite_strategies import (  # noqa: F401
        orb_pro,
        smc_liquidity,
        vwap_pro,
    )

    mods = {orb_pro.__name__, smc_liquidity.__name__, vwap_pro.__name__}
    assert len(mods) == 3

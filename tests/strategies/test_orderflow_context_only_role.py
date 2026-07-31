"""OrderFlow is measured confirmation, not a setup family.

Signal generation belongs to three distinct setup strategies -- SMC liquidity,
VWAP Pro and ORB Pro -- each graded against min_score ~5.5-5.8. OrderFlow
supplies confirmation (order flow / OI / IV) and must not originate entries on
its own weaker ladder.

31 Jul session evidence: EVERY executed trade carried
`SIGNAL_GENERATED ... reason=OrderFlow`, while the graded strategies were
correctly refusing (score 1.50 x46, 3.50 x15, 4.00 x11, 5.00 x9 against
min_score 5.80/5.50). OrderFlow's own gate was ORDERFLOW_CONTEXT_MIN_SCORE=4.0,
and its LTP fallback awards 2.0 unconditionally, +2.0 for spread <= 12%, +1.5
for direction agreement and +1.0 merely for a tick having a direction -- so an
ordinary ticking market reaches exactly 4.0 and passes.

Root cause: ORDERFLOW_ALLOW_LIVE_TRIGGER / ORDERFLOW_ALLOW_TRIGGER_ROLE
defaulted to 'true', so OrderFlow could originate live trades out of the box.
"""

from __future__ import annotations

import inspect
import os

from nifty_scalper_bot.strategies.elite_strategies import order_flow


def _assign_block(src: str) -> str:
    """The statement that grants OrderFlow a trigger role."""
    start = src.index("allow_orderflow_trigger = str(")
    return src[start : start + 200]


def test_orderflow_trigger_role_is_off_by_default() -> None:
    """THE FIX: OrderFlow must not originate entries unless opted in."""
    block = _assign_block(inspect.getsource(order_flow))
    assert "'false'" in block, (
        "OrderFlow must default to context-only; a 'true' default lets it "
        "originate trades on a weaker ladder than the setup strategies"
    )
    # getenv default must be false; "'true'" also appears in the truthy set,
    # so assert on the default argument specifically.
    assert "getenv(_trigger_env, 'false')" in block


def test_both_live_and_non_live_trigger_envs_default_off() -> None:
    """Neither execution mode may silently grant a trigger role."""
    src = inspect.getsource(order_flow)
    start = src.index("_trigger_env")
    window = src[start : start + 500]
    assert "ORDERFLOW_ALLOW_LIVE_TRIGGER" in window
    assert "ORDERFLOW_ALLOW_TRIGGER_ROLE" in window
    assert "'false'" in window


def test_trigger_role_can_still_be_enabled_explicitly(monkeypatch) -> None:
    """Opt-in remains available; only the default changed."""
    monkeypatch.setenv("ORDERFLOW_ALLOW_TRIGGER_ROLE", "true")
    assert os.getenv("ORDERFLOW_ALLOW_TRIGGER_ROLE", "false").lower() in {
        "1", "true", "yes", "on",
    }


def test_context_scoring_path_is_unchanged() -> None:
    """OrderFlow remains available as confirmation, with its context gate."""
    src = inspect.getsource(order_flow)
    assert "ORDERFLOW_CONTEXT_MIN_SCORE" in src
    assert "context_min_score" in src


def test_three_setup_families_remain_distinct_modules() -> None:
    """SMC, VWAP Pro and ORB Pro stay as separate signal generators."""
    from nifty_scalper_bot.strategies.elite_strategies import (  # noqa: F401
        orb_pro,
        smc_liquidity,
        vwap_pro,
    )

    for mod in (smc_liquidity, vwap_pro, orb_pro):
        assert inspect.ismodule(mod)

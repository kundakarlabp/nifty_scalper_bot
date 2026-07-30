"""ORDER_REJECTED must preserve its causal fields.

Post-#943 audit, P1. The warning carried only {"event", "symbol"}, so a
rejection could not be attributed to a safety gate, local validation,
margin/sizing or an actual broker failure, and could not be correlated with
the matching RUNNER_ORDER_MANAGER_REJECTED record. Three ORDER_REJECTED
events appeared in the 29 July session with no recoverable cause.

submit_reason, submit_details, broker_attempted and trace_id were all already
in scope at the emit site; they were simply discarded.
"""

from __future__ import annotations

import inspect

from nifty_scalper_bot.strategies.runner import StrategyRunner


def _emit_region() -> str:
    """Source around the ORDER_REJECTED emit site."""
    src = inspect.getsource(StrategyRunner)
    marker = '"ORDER_REJECTED by order_manager'
    assert marker in src, "ORDER_REJECTED emit site not found"
    start = src.index(marker)
    return src[start - 700 : start + 1400]


def test_order_rejected_carries_reason() -> None:
    """Without the reason a rejection class cannot be identified."""
    region = _emit_region()
    assert '"reason": submit_reason' in region


def test_order_rejected_carries_broker_attempted() -> None:
    """THE KEY DISCRIMINATOR: False means it never reached the broker."""
    region = _emit_region()
    assert '"broker_attempted": broker_attempted' in region


def test_order_rejected_carries_trace_id_for_correlation() -> None:
    """Needed to join with RUNNER_ORDER_MANAGER_REJECTED."""
    region = _emit_region()
    assert '"trace_id": trace_id' in region


def test_order_rejected_carries_structured_details() -> None:
    """Margin/sizing and kill-switch payloads must survive."""
    region = _emit_region()
    assert '"details": _reject_details' in region


def test_order_rejected_throttle_key_is_reason_scoped() -> None:
    """A 300s throttle keyed only on symbol hid distinct later causes."""
    region = _emit_region()
    assert 'f"runner_order_rejected_{base_symbol}_{submit_reason}"' in region


def test_order_rejected_still_identifies_both_symbols() -> None:
    """Underlying and the actual order symbol are both retained."""
    region = _emit_region()
    assert '"symbol": base_symbol' in region
    assert '"order_symbol": order_symbol' in region

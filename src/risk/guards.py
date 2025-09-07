"""Simple pre-trade risk guard helpers."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Deque

from src.broker.interface import OrderRequest, Side


@dataclass
class GuardConfig:
    """Configuration for :mod:`risk.guards`.  Zero values disable checks."""

    max_loss: Decimal = Decimal("0")
    max_position: int = 0
    max_exposure: Decimal = Decimal("0")
    max_orders_per_minute: int = 0


@dataclass
class GuardState:
    """Mutable risk state shared across guard evaluations."""

    realised_loss: Decimal = Decimal("0")
    position: int = 0
    exposure: Decimal = Decimal("0")
    order_ts: Deque[float] = field(default_factory=lambda: deque(maxlen=100))


def risk_check(order: OrderRequest, state: GuardState, cfg: GuardConfig) -> bool:
    """Return ``True`` if all configured guards pass for ``order``.

    The checks are intentionally lightweight and in‑memory, making them suitable
    for use inside tight event loops.
    """

    now = time.time()
    # --- order rate cap ---
    state.order_ts.append(now)
    while state.order_ts and now - state.order_ts[0] > 60:
        state.order_ts.popleft()
    if cfg.max_orders_per_minute and len(state.order_ts) > cfg.max_orders_per_minute:
        return False

    # --- daily loss cap ---
    if cfg.max_loss and state.realised_loss <= -abs(cfg.max_loss):
        return False

    # --- position & exposure caps ---
    new_pos = state.position + (order.qty if order.side is Side.BUY else -order.qty)
    if cfg.max_position and abs(new_pos) > cfg.max_position:
        return False
    notional = (order.price or Decimal("0")) * Decimal(order.qty)
    new_exposure = state.exposure + (
        notional if order.side is Side.BUY else -notional
    )
    if cfg.max_exposure and abs(new_exposure) > cfg.max_exposure:
        return False

    # Guards passed – update provisional state
    state.position = new_pos
    state.exposure = new_exposure
    return True


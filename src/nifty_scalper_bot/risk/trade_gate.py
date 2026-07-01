"""Trade gate before sizing/execution."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nifty_scalper_bot.core.signal_arbitrator import TradeDecision


@dataclass(slots=True)
class GateResult:
    allowed: bool
    reason: str
    details: dict[str, Any]


class TradeGate:
    def evaluate(self, decision: TradeDecision, state: dict[str, Any]) -> GateResult:
        if decision.action != 'BUY':
            return GateResult(False, 'not_buy_action', {})
        checks = [
            ('paused', False),
            ('emergency_halt', False),
            ('market_open', True),
            ('live_orders_armed', True),
        ]
        for key, expected in checks:
            if state.get(key, expected) != expected:
                return GateResult(False, key, state)
        max_daily_loss = abs(float(state.get('max_daily_loss', 0.0)))
        if max_daily_loss > 0 and float(state.get('daily_loss', 0.0)) <= -max_daily_loss:
            return GateResult(False, 'daily_loss_limit', state)
        if decision.symbol and decision.symbol in set(state.get('open_symbols', [])):
            return GateResult(False, 'duplicate_symbol_position', state)
        if float(decision.rr or 0.0) < 1.5:
            return GateResult(False, 'rr_low', state)
        return GateResult(True, 'allowed', state)

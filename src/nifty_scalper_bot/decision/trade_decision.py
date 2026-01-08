from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict
from nifty_scalper_bot.execution.fsm import FSMState


@dataclass(frozen=True)
class TradeDecision:
    decision_id: str
    strategy_id: str

    symbol: str
    expiry: str
    strike: int
    option_type: str  # CE / PE
    side: str         # BUY / SELL
    intent: str       # ENTRY / EXIT / HEDGE

    signal_score: float
    signal_type: str
    signal_time: datetime

    regime: Dict
    confidence: float

    idempotency_key: str
    current_state: FSMState
    next_state: FSMState

    allowed: bool
    reject_reason: Optional[str] = None

import hashlib
import uuid
from datetime import datetime
from nifty_scalper_bot.execution.fsm import FSMState
from .trade_decision import TradeDecision


class DecisionBuilder:

    @staticmethod
    def build(signal, regime, state):
        decision_id = str(uuid.uuid4())

        idempotency_key = hashlib.sha256(
            f"{signal.strategy_id}|{signal.symbol}|{signal.strike}|{signal.side}".encode()
        ).hexdigest()

        allowed = True
        reject_reason = None

        if regime is None:
            allowed = False
            reject_reason = "REGIME_MISSING"

        elif signal.confidence < 0.6:
            allowed = False
            reject_reason = "LOW_CONFIDENCE"

        next_state = FSMState.SIGNAL_QUALIFIED if allowed else state

        return TradeDecision(
            decision_id=decision_id,
            strategy_id=signal.strategy_id,
            symbol=signal.symbol,
            expiry=signal.expiry,
            strike=signal.strike,
            option_type=signal.option_type,
            side=signal.side,
            intent="ENTRY",
            signal_score=signal.score,
            signal_type=signal.type,
            signal_time=datetime.utcnow(),
            regime=regime,
            confidence=signal.confidence,
            idempotency_key=idempotency_key,
            current_state=state,
            next_state=next_state,
            allowed=allowed,
            reject_reason=reject_reason
        )

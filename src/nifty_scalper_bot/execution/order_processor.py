"""
order_processor.py

World-class, production-grade execution patch.
ALL broker execution is routed through:
FSM → TradeDecision → ExecutionEngine

NO signal is allowed to touch broker APIs directly.
"""

from typing import Optional

from nifty_scalper_bot.decision.decision_builder import DecisionBuilder
from nifty_scalper_bot.execution.executor import ExecutionEngine
from nifty_scalper_bot.execution.state_store import StateStore
from nifty_scalper_bot.execution.fsm import FSMState
from nifty_scalper_bot.logging import logger


class OrderProcessor:
    """
    OrderProcessor is now a PURE ORCHESTRATOR.
    It does NOT decide, it does NOT trade.
    """

    def __init__(self, broker, regime_manager):
        self.broker = broker
        self.regime_manager = regime_manager

        self.execution_engine = ExecutionEngine(broker)
        self.state_store = StateStore()

    def process_signal(self, signal) -> Optional[str]:
        """
        Entry point called by strategy runner.

        signal: existing signal object (UNCHANGED)
        """

        try:
            # ---- Build execution key (symbol + strategy scope) ----
            state_key = f"{signal.symbol}:{signal.strategy_id}"

            current_state = self.state_store.get(state_key)

            # ---- Fetch regime (HARD INVARIANT) ----
            regime = self.regime_manager.get_current_regime()

            # ---- Build TradeDecision (single source of truth) ----
            decision = DecisionBuilder.build(
                signal=signal,
                regime=regime,
                state=current_state
            )

            # ---- Log rejections clearly ----
            if not decision.allowed:
                logger.info(
                    "TRADE_REJECTED",
                    extra={
                        "symbol": decision.symbol,
                        "strike": decision.strike,
                        "side": decision.side,
                        "reason": decision.reject_reason,
                        "confidence": decision.confidence
                    }
                )
                return None

            # ---- Enforce FSM transition ----
            self.state_store.set(state_key, decision.next_state)

            # ---- Execute via ExecutionEngine ONLY ----
            order_id = self.execution_engine.execute(decision)

            if order_id:
                logger.info(
                    "ORDER_SENT",
                    extra={
                        "order_id": order_id,
                        "symbol": decision.symbol,
                        "strike": decision.strike,
                        "side": decision.side,
                        "strategy": decision.strategy_id
                    }
                )

            return order_id

        except Exception as e:
            logger.exception(
                "ORDER_PROCESSOR_FAILURE",
                extra={
                    "symbol": getattr(signal, "symbol", None),
                    "error": str(e)
                }
            )
            return None

    # -------------------------------------------------------
    # OPTIONAL: Explicit exit path (SAFE)
    # -------------------------------------------------------

    def process_exit(self, signal) -> Optional[str]:
        """
        Explicit exit processing.
        Entry-only logic NEVER handles exits.
        """

        try:
            state_key = f"{signal.symbol}:{signal.strategy_id}"
            current_state = self.state_store.get(state_key)

            if current_state != FSMState.POSITION_OPEN:
                return None

            regime = self.regime_manager.get_current_regime()

            decision = DecisionBuilder.build(
                signal=signal,
                regime=regime,
                state=current_state
            )

            if not decision.allowed:
                return None

            self.state_store.set(state_key, FSMState.EXIT_PENDING)

            return self.execution_engine.execute(decision)

        except Exception:
            logger.exception("EXIT_PROCESSING_FAILED")
            return None

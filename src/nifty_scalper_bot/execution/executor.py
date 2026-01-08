from nifty_scalper_bot.execution.fsm import is_transition_allowed
from nifty_scalper_bot.execution.state_store import StateStore


class ExecutionEngine:
    def __init__(self, broker):
        self.broker = broker
        self.state_store = StateStore()
        self.seen_keys = set()

    def execute(self, decision):
        key = f"{decision.symbol}:{decision.strategy_id}"

        if decision.idempotency_key in self.seen_keys:
            return

        current_state = self.state_store.get(key)

        if not decision.allowed:
            return

        if not is_transition_allowed(current_state, decision.next_state):
            return

        self.state_store.set(key, decision.next_state)

        # --- SINGLE BROKER ENTRY POINT ---
        order_id = self.broker.place_order(
            symbol=decision.symbol,
            strike=decision.strike,
            option_type=decision.option_type,
            side=decision.side
        )

        self.seen_keys.add(decision.idempotency_key)
        return order_id

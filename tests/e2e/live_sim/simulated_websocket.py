from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SubscriptionProof:
    symbol: str
    token: int
    generation: int | None
    state: str


class SimulatedWebSocket:
    is_simulated_adapter = True
    """Production-stream seam for deterministic subscription lifecycle tests."""

    def __init__(self, market_data: Any, recorder: Any | None = None) -> None:
        self.market_data = market_data
        self.recorder = recorder
        self.transitions: list[SubscriptionProof] = []
        self._callbacks: dict[str, Any] = {}
        self.connected = False


    def set_callbacks(self, **callbacks: Any) -> None:
        existing = getattr(self, "_callbacks", {})
        existing.update(callbacks)
        self._callbacks = existing

    def connect(self) -> None:
        self.connected = True

        callback = (
            self._callbacks.get("on_connect")
            or self._callbacks.get("on_open")
        )

        if callback is not None:
            result = callback()
            if inspect.isawaitable(result):
                asyncio.get_event_loop().run_until_complete(result)

    def publish_tick(self, tick: dict[str, Any]) -> None:
        if not self.connected:
            raise RuntimeError("simulated websocket is not connected")

        callback = (
            self._callbacks.get("on_ticks")
            or self._callbacks.get("on_tick")
        )

        if callback is None:
            raise RuntimeError(
                "production websocket tick callback was not registered"
            )

        try:
            result = callback([tick])
        except TypeError:
            result = callback(tick)

        if inspect.isawaitable(result):
            loop = asyncio.get_event_loop()
            loop.run_until_complete(result)

    def request(self, symbol: str, token: int) -> SubscriptionProof:
        self.market_data.register_symbol(symbol, token)
        self.market_data.request_token_subscription(token, symbol=symbol)
        return self._record(symbol, token, "REQUESTED")

    def dispatch(self, symbol: str, token: int) -> SubscriptionProof:
        with self.market_data._lock:  # noqa: SLF001 - adapter owns simulated ACK
            self.market_data._tracked_symbols.add(symbol)  # noqa: SLF001
            self.market_data._dispatched_subscriptions.add(int(token))  # noqa: SLF001
        return self._record(symbol, token, "DISPATCHED")

    def confirm(self, symbol: str, token: int) -> SubscriptionProof:
        with self.market_data._lock:  # noqa: SLF001 - adapter owns simulated ACK
            self.market_data._tracked_symbols.add(symbol)  # noqa: SLF001
            self.market_data._dispatched_subscriptions.add(int(token))  # noqa: SLF001
            self.market_data._confirmed_subscriptions.add(int(token))  # noqa: SLF001
        return self._record(symbol, token, "CONFIRMED")

    def mark_first_current_generation_tick(
        self, symbol: str, token: int
    ) -> SubscriptionProof:
        with self.market_data._lock:  # noqa: SLF001 - adapter owns simulated tick ACK
            generation = self.market_data._symbol_subscription_generation.get(
                symbol
            )  # noqa: SLF001
            self.market_data._symbol_first_tick_generation[symbol] = (
                generation  # noqa: SLF001
            )
            self.market_data._last_valid_live_tick_mono[symbol] = (
                time.monotonic()
            )  # noqa: SLF001
        return self._record(symbol, token, "ACTIVE")

    def activate(self, symbol: str, token: int) -> SubscriptionProof:
        self.request(symbol, token)
        self.dispatch(symbol, token)
        self.confirm(symbol, token)
        return self.mark_first_current_generation_tick(symbol, token)

    def _record(self, symbol: str, token: int, state: str) -> SubscriptionProof:
        generation = getattr(
            self.market_data, "_symbol_subscription_generation", {}
        ).get(symbol)
        proof = SubscriptionProof(symbol, int(token), generation, state)
        self.transitions.append(proof)
        if self.recorder is not None:
            self.recorder.record(
                "LIVE_SYMBOL_ACTIVATION",
                symbol,
                token=int(token),
                subscription_generation=generation,
                subscription_state=state,
            )
        return proof

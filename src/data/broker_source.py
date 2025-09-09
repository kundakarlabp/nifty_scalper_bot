from __future__ import annotations

"""Data source that relays ticks from a :class:`Broker`."""

from typing import Callable, Optional, Sequence

from src.broker.interface import Broker, Tick


class BrokerDataSource:
    """Minimal adapter exposing a broker's tick stream.

    The orchestrator can register a callback via :meth:`set_tick_callback` or its
    alias :meth:`on_tick`, subscribe to instruments, and start the stream.
    """

    def __init__(self, broker: Broker) -> None:
        self.broker = broker
        self._tick_cb: Optional[Callable[[Tick], None]] = None
        self._subscriptions: Sequence[int] = []

    def set_tick_callback(self, cb: Callable[[Tick], None]) -> None:
        """Set the callback invoked for each tick."""
        self._tick_cb = cb

    def on_tick(
        self, cb: Callable[[Tick], None]
    ) -> None:  # pragma: no cover - simple alias
        """Alias for :meth:`set_tick_callback`."""
        self.set_tick_callback(cb)

    def subscribe(self, instruments: Sequence[int]) -> None:
        """Subscribe to tick updates for ``instruments``."""
        if self._tick_cb is None:
            raise RuntimeError("tick callback not set")
        self._subscriptions = list(instruments)
        self.broker.subscribe_ticks(self._subscriptions, self._handle_tick)

    def start(self) -> None:
        """Connect to the broker and resubscribe if needed."""
        if not self.broker.is_connected():
            self.broker.connect()
        if self._subscriptions and self._tick_cb is not None:
            self.broker.subscribe_ticks(self._subscriptions, self._handle_tick)
        # Register reconnect hooks if the broker exposes them
        on_reconnect = getattr(self.broker, "on_reconnect", None)
        if callable(on_reconnect):
            on_reconnect(lambda: self.subscribe(self._subscriptions))
        on_disconnect = getattr(self.broker, "on_disconnect", None)
        if callable(on_disconnect):
            on_disconnect(self.broker.connect)

    def stop(self) -> None:
        """Disconnect from the broker if it exposes ``disconnect``."""
        disconnect = getattr(self.broker, "disconnect", None)
        if callable(disconnect):
            try:
                disconnect()
            except Exception:  # pragma: no cover - defensive
                pass

    # ------------------------------------------------------------------
    def _handle_tick(self, tick: Tick) -> None:
        cb = self._tick_cb
        if cb is None:
            return
        try:
            cb(tick)
        except Exception:
            # Swallow exceptions to avoid blocking broker threads
            pass

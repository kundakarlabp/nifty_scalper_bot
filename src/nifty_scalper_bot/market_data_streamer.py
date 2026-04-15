"""WebSocket streamer adapter with MarketState synchronization."""

from __future__ import annotations

from collections.abc import Sequence
import time
import asyncio
from typing import Any

from nifty_scalper_bot.data.market_state import MarketState
from nifty_scalper_bot.core.message_bus import Message, MessageType
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


class MarketDataStreamer:
    """Manage WS callbacks and token subscriptions with deterministic state sync."""

    def __init__(self, websocket: Any, market_state: MarketState) -> None:
        """Initialize streamer. Args: websocket/market_state. Returns: none. Raises: none."""

        self._ws = websocket
        self._market_state = market_state
        self._subscribed_tokens: set[int] = set()
        self._last_heartbeat: float | None = None
        self.bus = None

    @property
    def last_heartbeat(self) -> float | None:
        """Return last heartbeat timestamp. Args: none. Returns: ts|None. Raises: none."""

        return self._last_heartbeat

    def on_ticks(self, ticks: Sequence[dict[str, Any]]) -> None:
        """Update market state from websocket ticks. Args: ticks. Returns: none. Raises: none."""

        now = time.time()
        self._last_heartbeat = now
        for tick in ticks:
            try:
                token = int(tick.get("instrument_token"))
                price = float(tick.get("last_price") or tick.get("ltp"))
                if token == 256265:
                    self._market_state.set_spot_ltp(price, source="ws")
                self._market_state.update_tick(token, price, source="ws")

                if self.bus is not None:
                    msg = Message(
                        type=MessageType.TICK,
                        timestamp=now,
                        data={
                            "token": token,
                            "ltp": price,
                            "ts": tick.get("exchange_timestamp", now)
                        },
                        source="market_data_streamer"
                    )
                    asyncio.create_task(self.bus.publish(msg))
            except Exception as e:
                LOGGER.exception("Failure in MarketDataStreamer.on_ticks: %s", e)

    def on_error(self, error: Exception) -> None:
        """Handle websocket error. Args: error. Returns: none. Raises: none."""

        LOGGER.exception("WebSocket error, reconnecting: %s", error)
        self._attempt_reconnect()

    def on_close(self) -> None:
        """Handle websocket close. Args: none. Returns: none. Raises: none."""

        LOGGER.warning("WebSocket closed, reconnecting")
        self._attempt_reconnect()

    def update_subscription(self, tokens: Sequence[int]) -> None:
        """Synchronize websocket subscription set. Args: tokens. Returns: none. Raises: none."""

        new_tokens = {int(token) for token in tokens}
        old_tokens = set(self._subscribed_tokens)
        if not new_tokens and not old_tokens:
            return
        remove_tokens = sorted(old_tokens - new_tokens)
        add_tokens = sorted(new_tokens - old_tokens)
        try:
            if remove_tokens:
                self._ws.unsubscribe(remove_tokens)
            if add_tokens:
                self._ws.subscribe(add_tokens)
                mode_ltp = getattr(self._ws, "MODE_LTP", None)
                if mode_ltp is not None:
                    self._ws.set_mode(mode_ltp, add_tokens)
            self._subscribed_tokens = new_tokens
        except Exception as e:
            LOGGER.exception("Failure in MarketDataStreamer.update_subscription: %s", e)

    def _attempt_reconnect(self) -> None:
        """Attempt websocket reconnect safely. Args: none. Returns: none. Raises: none."""

        try:
            reconnect = getattr(self._ws, "connect", None)
            if callable(reconnect):
                reconnect()
        except Exception as e:
            LOGGER.exception("Failure in MarketDataStreamer._attempt_reconnect: %s", e)

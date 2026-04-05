"""Zerodha KiteTicker WebSocket stream with warmup guard."""

from __future__ import annotations

import logging
import time as time_module
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data.tick_hub import TickHub
    from ..config.settings import StreamingSettings, BrokerSettings

log = logging.getLogger(__name__)


class WebSocketStream:
    """Connects to Zerodha KiteTicker and feeds ticks into TickHub."""

    def __init__(
        self,
        tick_hub: "TickHub",
        broker_settings: "BrokerSettings",
        streaming_settings: "StreamingSettings",
        token_map: dict[int, str],
    ) -> None:
        self._hub = tick_hub
        self._broker = broker_settings
        self._stream = streaming_settings
        self._token_map = token_map
        self._ticker = None
        self._running = False
        self._reconnect_count = 0
        self._tick_counts: dict[int, int] = {}
        self._connected = False

    def start(self) -> None:
        try:
            from kiteconnect import KiteTicker  # type: ignore[import]
            self._ticker = KiteTicker(
                api_key=self._broker.api_key.get_secret_value(),
                access_token=self._broker.access_token.get_secret_value(),
            )
            self._ticker.on_connect = self._on_connect
            self._ticker.on_ticks = self._on_ticks
            self._ticker.on_close = self._on_close
            self._ticker.on_error = self._on_error
            self._running = True
            self._ticker.connect(threaded=True)
            log.info("WebSocket stream started")
        except Exception as e:
            log.error("WebSocket start failed: %s", e)
            raise

    def stop(self) -> None:
        self._running = False
        if self._ticker:
            try:
                self._ticker.close()
            except Exception:
                pass

    def subscribe(self, tokens: list[int]) -> None:
        if self._ticker and tokens:
            self._ticker.subscribe(tokens)
            self._ticker.set_mode(self._ticker.MODE_FULL, tokens)
            # Register symbols for warmup tracking
            for token in tokens:
                sym = self._token_map.get(token, str(token))
                self._hub.register_symbol(sym)

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _on_connect(self, ws, response) -> None:
        self._connected = True
        self._reconnect_count = 0
        tokens = list(self._token_map.keys())
        if tokens:
            ws.subscribe(tokens)
            ws.set_mode(ws.MODE_FULL, tokens)
        log.info("WebSocket connected, subscribed %d tokens", len(tokens))

    def _on_ticks(self, ws, ticks: list[dict]) -> None:
        from ..data.tick_hub import Tick
        for raw in ticks:
            token = raw.get("instrument_token", 0)
            sym = self._token_map.get(token, str(token))
            count = self._tick_counts.get(token, 0) + 1
            self._tick_counts[token] = count
            is_live = count > self._stream.warmup_ticks

            tick = Tick(
                symbol=sym,
                instrument_token=token,
                ltp=raw.get("last_price", 0.0),
                bid=raw.get("depth", {}).get("buy", [{}])[0].get("price", 0.0),
                ask=raw.get("depth", {}).get("sell", [{}])[0].get("price", 0.0),
                volume=raw.get("volume", 0),
                oi=raw.get("oi", 0),
                timestamp=datetime.now(timezone.utc),
                is_backfill=not is_live,
            )
            self._hub.on_tick_sync(tick)

    def _on_close(self, ws, code, reason) -> None:
        self._connected = False
        log.warning("WebSocket closed: %s %s", code, reason)

    def _on_error(self, ws, code, reason) -> None:
        log.error("WebSocket error: %s %s", code, reason)

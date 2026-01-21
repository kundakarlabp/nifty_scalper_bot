# src/nifty_scalper_bot/streaming/kite_ticker_streamer.py

from kiteconnect import KiteTicker
import threading
import time
import logging

LOGGER = logging.getLogger(__name__)

class KiteTickerStreamer:
    def __init__(self, broker, data_hub, tokens: list[int]):
        self._broker = broker
        self._data_hub = data_hub
        self._tokens = tokens
        self._ticker = None
        self._thread = None
        self._last_tick_ts = time.monotonic()

    def start(self):
        self._ticker = KiteTicker(
            self._broker.api_key,
            self._broker.access_token
        )

        self._ticker.on_ticks = self._on_ticks
        self._ticker.on_connect = self._on_connect
        self._ticker.on_close = self._on_close
        self._ticker.on_error = self._on_error

        self._thread = threading.Thread(
            target=self._ticker.connect,
            kwargs={"threaded": True},
            daemon=True,
        )
        self._thread.start()

    def _on_connect(self, ws, response):
        LOGGER.info("✅ KiteTicker connected")
        ws.subscribe(self._tokens)
        ws.set_mode(ws.MODE_FULL, self._tokens)

    def _on_ticks(self, ws, ticks):
        now = int(time.time() * 1000)
        self._last_tick_ts = time.monotonic()

        for tick in ticks:
            token = tick.get("instrument_token")
            symbol = self._broker.resolve_token(token)
            if not symbol:
                continue

            tick["symbol"] = symbol
            tick["timestamp"] = now
            tick["source"] = "ws"

            self._data_hub.store_quote(symbol, tick, source="ws")

    def _on_close(self, ws, code, reason):
        LOGGER.warning(f"KiteTicker closed: {code} {reason}")

    def _on_error(self, ws, code, reason):
        LOGGER.error(f"KiteTicker error: {code} {reason}")

    def last_tick_age(self):
        return time.monotonic() - self._last_tick_ts

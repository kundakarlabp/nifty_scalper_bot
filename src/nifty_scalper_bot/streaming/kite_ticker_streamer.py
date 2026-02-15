# src/nifty_scalper_bot/streaming/kite_ticker_streamer.py
# FIXED VERSION - Added resolver parameter and _resolve_token_to_symbol method

from kiteconnect import KiteTicker
import threading
import time
import logging

LOGGER = logging.getLogger(__name__)


class KiteTickerStreamer:
    """WebSocket-based market data streamer using Zerodha KiteTicker."""
    
    def __init__(self, broker, data_hub, tokens: list[int], resolver=None):
        """
        Initialize KiteTicker streamer.
        
        Args:
            broker: ZerodhaKiteClient or broker wrapper with api_key and access_token
            data_hub: DataHub instance for storing quotes
            tokens: List of instrument tokens to subscribe
            resolver: Optional InstrumentResolver for token-to-symbol mapping
        """
        self._broker = broker
        self._data_hub = data_hub
        self._tokens = tokens
        self._resolver = resolver
        self._ticker = None
        self._thread = None
        self._last_tick_ts = time.monotonic()

    def start(self):
        if self._ticker:
            LOGGER.warning("WebSocket already initialized")
            return

        self._ticker = KiteTicker(
            self._broker.api_key,
            self._broker.access_token
        )

        self._ticker.on_ticks = self._on_ticks
        self._ticker.on_connect = self._on_connect
        self._ticker.on_close = self._on_close
        self._ticker.on_error = self._on_error

        # ✅ FIX: Let KiteTicker manage its own internal thread
        self._ticker.connect(threaded=True)

        LOGGER.info("🚀 KiteTicker WebSocket started (threaded mode)")

    def _on_connect(self, ws, response):
        """Handle WebSocket connection."""
        LOGGER.info("✅ KiteTicker connected")
        if self._tokens:
            ws.subscribe(self._tokens)
            ws.set_mode(ws.MODE_FULL, self._tokens)
            LOGGER.info(f"📡 Subscribed to {len(self._tokens)} tokens")

    def _on_ticks(self, ws, ticks):
        """Handle incoming ticks from WebSocket with Normalization + Active Ingestion."""
        now = int(time.time() * 1000)
        self._last_tick_ts = time.monotonic()

        for tick in ticks:
            token = tick.get("instrument_token")
            if not token:
                continue
                
            symbol = self._resolve_token_to_symbol(token)
            if not symbol:
                continue

            # ===========================================================
            # 1️⃣ DATA NORMALIZATION (Zerodha -> Bot Canonical)
            # ===========================================================
            
            # Price
            if "last_price" in tick:
                tick["ltp"] = tick["last_price"]
            
            # Volume (Try all Zerodha keys)
            # This fixes "Vol=0" in logs
            vol = tick.get("volume_traded") or tick.get("last_traded_quantity")
            if vol is not None:
                tick["volume"] = vol
            else:
                tick["volume"] = 0
                
            # VWAP (Average Traded Price)
            # This fixes "VWAP=0.0" in logs
            if "average_price" in tick:
                tick["vwap"] = tick["average_price"]
            else:
                tick["vwap"] = 0.0
                
            # Open Interest (Support dual keys)
            if "oi" in tick:
                tick["open_interest"] = tick["oi"]
            
            # Depth Flattening
            if "depth" in tick:
                buy_depth = tick["depth"].get("buy", [])
                sell_depth = tick["depth"].get("sell", [])
                if buy_depth:
                    tick["best_bid"] = buy_depth[0].get("price", 0.0)
                    tick["best_bid_qty"] = buy_depth[0].get("quantity", 0)
                if sell_depth:
                    tick["best_ask"] = sell_depth[0].get("price", 0.0)
                    tick["best_ask_qty"] = sell_depth[0].get("quantity", 0)

            # Metadata
            tick["symbol"] = symbol
            tick["timestamp"] = now
            tick["source"] = "ws"

            # ===========================================================
            # 2️⃣ INGESTION PATH FIX (The "Smoking Gun")
            # ===========================================================
            # WAS: self._data_hub.store_quote(symbol, tick, source="ws")
            # FIX: Use ingest_tick_sync to trigger MessageBus, Freshness, and Strategies
            
            if self._data_hub:
                try:
                    # Prefer the sync wrapper if available (handles async loop dispatch)
                    if hasattr(self._data_hub, "ingest_tick_sync"):
                        self._data_hub.ingest_tick_sync(tick)
                    else:
                        # Fallback for safety
                        self._data_hub.store_quote(symbol, tick, source="ws")
                except Exception as e:
                    LOGGER.debug(f"DataHub ingestion failed: {e}")

    def _on_close(self, ws, code, reason):
        """Handle WebSocket close."""
        LOGGER.warning(f"KiteTicker closed: {code} {reason}")

    def _on_error(self, ws, code, reason):
        """Handle WebSocket error."""
        LOGGER.error(f"KiteTicker error: {code} {reason}")

    def last_tick_age(self) -> float:
        """Return seconds since last tick was received."""
        return time.monotonic() - self._last_tick_ts

    def _resolve_token_to_symbol(self, token: int) -> str | None:
        """
        Resolve instrument token to strict EXCHANGE:SYMBOL format.
        Ensures strategies can match subscriptions to incoming ticks.
        """
        try:
            # 1. Fast Path: Known Canonical Tokens (Indices)
            # Checking this first is an optimization to avoid overhead for common symbols.
            CANONICAL_TOKENS = {
                256265: "NSE:NIFTY 50",
                260105: "NSE:NIFTY BANK",
            }
            if token in CANONICAL_TOKENS:
                return CANONICAL_TOKENS[token]

            # 2. Try Resolver Lookup (Preferred - Source of Truth)
            # This is safer because it retrieves the specific 'exchange' field from metadata.
            if self._resolver and hasattr(self._resolver, "lookup"):
                info = self._resolver.lookup(token)
                if info:
                    exchange = info.get("exchange", "NFO")
                    sym = info.get("tradingsymbol") or info.get("symbol")
                    if sym:
                        return f"{exchange}:{sym}"

            # 3. Fallback: Format String (Resolver or Broker)
            # If lookup fails, try to get the string representation.
            raw_sym = None
            
            # Try resolver first
            if self._resolver and hasattr(self._resolver, "format_token_as_symbol"):
                raw_sym = self._resolver.format_token_as_symbol(token)
            
            # Try broker second
            if (not raw_sym or raw_sym == str(token)) and hasattr(self._broker, "format_token_as_symbol"):
                raw_sym = self._broker.format_token_as_symbol(token)

            # 4. Normalization Fix (Critical)
            # If we got a valid symbol string, ensure it has an exchange prefix.
            if raw_sym and raw_sym != str(token):
                if ":" in raw_sym:
                    return raw_sym
                
                # If exchange is missing, force NFO (since NSE indices are caught in Step 1)
                return f"NFO:{raw_sym}"

        except Exception as e:
            LOGGER.debug(f"Token resolution failed for {token}: {e}")
        
        return None
        
    def subscribe(self, tokens: list[int]) -> None:
        """Subscribe to additional tokens."""
        new_tokens = [t for t in tokens if t not in self._tokens]
        if new_tokens:
            self._tokens.extend(new_tokens)
            if self._ticker and hasattr(self._ticker, "subscribe"):
                try:
                    self._ticker.subscribe(new_tokens)
                    self._ticker.set_mode(self._ticker.MODE_FULL, new_tokens)
                    LOGGER.info(f"📡 Subscribed to {len(new_tokens)} additional tokens")
                except Exception as e:
                    LOGGER.error(f"Failed to subscribe tokens: {e}")

    def unsubscribe(self, tokens: list[int]) -> None:
        """Unsubscribe from tokens."""
        for t in tokens:
            if t in self._tokens:
                self._tokens.remove(t)
        if self._ticker and hasattr(self._ticker, "unsubscribe"):
            try:
                self._ticker.unsubscribe(tokens)
            except Exception as e:
                LOGGER.error(f"Failed to unsubscribe tokens: {e}")

    def stop(self) -> None:
        """Stop the KiteTicker connection."""
        if self._ticker:
            try:
                self._ticker.close()
            except Exception as e:
                LOGGER.debug(f"Error closing ticker: {e}")
        LOGGER.info("🛑 KiteTicker stopped")

    def is_running(self) -> bool:
        """Check if the ticker is running."""
        return self._thread is not None and self._thread.is_alive()

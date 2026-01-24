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
        """Start the KiteTicker WebSocket connection."""
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
        LOGGER.info("🚀 KiteTicker WebSocket starting...")

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
        Resolve instrument token to exchange:symbol format.
        
        Args:
            token: Instrument token (integer)
            
        Returns:
            Symbol string like 'NSE:NIFTY 50' or 'NFO:NIFTY26JAN25200CE', or None
        """
        try:
            # 1. Try resolver if available (preferred)
            if self._resolver is not None:
                # Method 1: format_token_as_symbol
                if hasattr(self._resolver, "format_token_as_symbol"):
                    result = self._resolver.format_token_as_symbol(token)
                    if result and result != str(token):
                        return result
                
                # Method 2: lookup
                if hasattr(self._resolver, "lookup"):
                    info = self._resolver.lookup(token)
                    if info:
                        exchange = info.get("exchange", "NFO")
                        sym = info.get("symbol") or info.get("tradingsymbol")
                        if sym:
                            return f"{exchange}:{sym}"
            
            # 2. Try broker's resolver if available
            if hasattr(self._broker, "_resolver") and self._broker._resolver:
                resolver = self._broker._resolver
                if hasattr(resolver, "format_token_as_symbol"):
                    result = resolver.format_token_as_symbol(token)
                    if result and result != str(token):
                        return result
            
            # 3. Try broker's format_token_as_symbol directly
            if hasattr(self._broker, "format_token_as_symbol"):
                result = self._broker.format_token_as_symbol(token)
                if result and result != str(token):
                    return result
            
            # 4. Canonical tokens fallback
            CANONICAL_TOKENS = {
                256265: "NSE:NIFTY 50",
                260105: "NSE:NIFTY BANK",
            }
            if token in CANONICAL_TOKENS:
                return CANONICAL_TOKENS[token]
                
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

"""Single source of truth for instrument token/symbol mappings.

InstrumentManager loads NFO instruments directly from the broker API and
maintains a bi-directional token↔symbol map.  Every downstream component
(market data, WebSocket subscription, polling, hydration) must obtain tokens
exclusively through this manager — never by constructing option symbol strings
manually.

Usage::

    mgr = InstrumentManager(kite_client)
    mgr.load()

    token = mgr.get_token("NIFTY26APR25600CE")  # raises RuntimeError if missing
    symbol = mgr.get_symbol(12345678)            # returns None if unknown

Designed to replace ad-hoc resolver calls that silently returned None and
produced `instrument_resolver_no_token` log errors downstream.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

LOGGER = logging.getLogger("nifty_scalper_bot.core.instrument_manager")


class InstrumentManager:
    """Token-first instrument map loaded directly from broker NFO dump.

    Thread-safe: all internal caches are protected by an RLock.
    """

    def __init__(self, kite: Any) -> None:
        """Args: kite – broker client with an .instruments(exchange) method.
        Returns: None. Raises: TypeError when kite is None.
        """
        if kite is None:
            raise TypeError("InstrumentManager requires a non-None broker client")
        self._kite = kite
        self._token_map: dict[str, int] = {}   # tradingsymbol.upper() → token
        self._symbol_map: dict[int, str] = {}  # token → tradingsymbol (bare)
        self._exchange_map: dict[int, str] = {}  # token → exchange
        self._lock = threading.RLock()
        self._loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Fetch NFO instruments from broker and populate internal maps.

        Args: none.
        Returns: None.
        Raises: RuntimeError when no NIFTY instruments are found.
        """
        LOGGER.info("InstrumentManager: loading NFO instruments from broker…")
        raw = self._kite.instruments("NFO")

        count = 0
        with self._lock:
            self._token_map.clear()
            self._symbol_map.clear()
            self._exchange_map.clear()

            for inst in raw:
                name = str(inst.get("name", "")).upper()
                if name != "NIFTY":
                    continue

                tradingsymbol = str(inst.get("tradingsymbol") or "").strip()
                token_raw = inst.get("instrument_token")
                exchange = str(inst.get("exchange") or "NFO").strip().upper()

                if not tradingsymbol or token_raw is None:
                    continue

                try:
                    token = int(token_raw)
                except (TypeError, ValueError):
                    continue

                key = tradingsymbol.upper()
                self._token_map[key] = token
                self._token_map[f"{exchange}:{key}"] = token
                self._symbol_by_token_set(token, tradingsymbol, exchange)
                count += 1

            self._loaded = True

        if count == 0:
            raise RuntimeError(
                "[FATAL] InstrumentManager: no NIFTY instruments found in NFO dump. "
                "Check broker authentication and instrument endpoint."
            )

        LOGGER.info(
            "InstrumentManager: loaded %d NIFTY instruments from NFO",
            count,
            extra={"event": "instrument_manager_loaded", "count": count},
        )

    def get_token(self, symbol: str) -> int:
        """Return the broker instrument token for *symbol*.

        Args: symbol – bare tradingsymbol (e.g. 'NIFTY26APR25600CE') or
                       exchange-qualified (e.g. 'NFO:NIFTY26APR25600CE').
        Returns: integer instrument token.
        Raises: RuntimeError when the symbol cannot be resolved.
        """
        with self._lock:
            key = str(symbol).strip().upper()
            token = self._token_map.get(key)
            if token is None:
                # try without exchange prefix
                bare = key.split(":", 1)[-1]
                token = self._token_map.get(bare)
            if token is None:
                raise RuntimeError(
                    f"[FATAL] InstrumentManager: token not found for '{symbol}'. "
                    "Call load() first or check that the instrument exists in NFO dump."
                )
            return token

    def get_symbol(self, token: int) -> Optional[str]:
        """Return the tradingsymbol for *token*, or None when unknown.

        Args: token – integer instrument token.
        Returns: tradingsymbol string or None.
        Raises: None.
        """
        with self._lock:
            return self._symbol_map.get(int(token))

    def get_exchange(self, token: int) -> str:
        """Return exchange string for *token* (defaults to 'NFO').

        Args: token – integer instrument token.
        Returns: exchange string.
        Raises: None.
        """
        with self._lock:
            return self._exchange_map.get(int(token), "NFO")

    def all_tokens(self) -> list[int]:
        """Return sorted list of all known instrument tokens.

        Args: none. Returns: list[int]. Raises: None.
        """
        with self._lock:
            return sorted(self._symbol_map.keys())

    def is_loaded(self) -> bool:
        """Return True when load() has completed successfully.

        Args: none. Returns: bool. Raises: None.
        """
        return self._loaded

    def size(self) -> int:
        """Return number of NIFTY instruments currently tracked.

        Args: none. Returns: int. Raises: None.
        """
        with self._lock:
            return len(self._symbol_map)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _symbol_by_token_set(
        self, token: int, tradingsymbol: str, exchange: str
    ) -> None:
        """Populate reverse maps. Must be called under self._lock."""
        self._symbol_map[token] = tradingsymbol
        self._exchange_map[token] = exchange

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
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger("nifty_scalper_bot.core.instrument_manager")

# Well-known index tokens
_WELL_KNOWN_TOKENS: Dict[str, int] = {
    "NIFTY": 256265,
    "BANKNIFTY": 260105,
    "NSE:NIFTY": 256265,
    "NSE:BANKNIFTY": 260105,
}


def _atm_strike_for_spot(spot: float, step: int) -> int:
    """Return the nearest ATM strike for a spot price."""
    if spot <= 0 or step <= 0:
        return 0
    return int(round(spot / step) * step)


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
        self._lot_size_map: dict[int, int] = {}  # token → lot_size
        self._instrument_data: dict[int, dict] = {}  # token → full instrument dict
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
            self._lot_size_map.clear()
            self._instrument_data.clear()

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
                
                # Store lot size - NIFTY options now have lot size of 65
                lot_size = inst.get("lot_size")
                if lot_size:
                    try:
                        self._lot_size_map[token] = int(lot_size)
                    except (TypeError, ValueError):
                        pass
                
                # Store full instrument data for lookup
                self._instrument_data[token] = dict(inst)
                
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

    def get_lot_size(self, token_or_symbol: int | str) -> Optional[int]:
        """Get lot size for a token or symbol.
        
        Args: token_or_symbol – either integer token or string symbol.
        Returns: lot size integer or None if not found.
        Raises: None.
        """
        with self._lock:
            # If it's a symbol, convert to token first
            if isinstance(token_or_symbol, str):
                key = str(token_or_symbol).strip().upper()
                token = self._token_map.get(key)
                if token is None:
                    # Try without exchange prefix
                    bare = key.split(":", 1)[-1]
                    token = self._token_map.get(bare)
                if token is None:
                    return None
            else:
                token = int(token_or_symbol)
            
            # Return lot size - NIFTY options should return 65 (current lot size)
            return self._lot_size_map.get(token)

    def lookup(self, token_or_symbol: int | str) -> Optional[dict]:
        """Lookup full instrument data by token or symbol.
        
        Args: token_or_symbol – either integer token or string symbol.
        Returns: Full instrument dict from broker or None if not found.
        Raises: None.
        """
        with self._lock:
            # If it's a symbol, convert to token first
            if isinstance(token_or_symbol, str):
                key = str(token_or_symbol).strip().upper()
                token = self._token_map.get(key)
                if token is None:
                    # Try without exchange prefix
                    bare = key.split(":", 1)[-1]
                    token = self._token_map.get(bare)
                if token is None:
                    return None
            else:
                token = int(token_or_symbol)
            
            return self._instrument_data.get(token)

    def get_instruments_by_expiry(self, expiry_date: date) -> List[Dict[str, Any]]:
        """Get all instruments expiring on a specific date.

        Args: expiry_date – the expiry date to filter by.
        Returns: List of instrument dicts expiring on that date.
        Raises: None.
        """
        result = []
        with self._lock:
            for token, inst_data in self._instrument_data.items():
                inst_expiry = self._parse_expiry(inst_data.get("expiry"))
                if inst_expiry == expiry_date:
                    result.append(dict(inst_data))
        return result

    def get_weekly_expiry_dates(self, num_weeks: int = 4) -> List[date]:
        """Get upcoming weekly expiry dates for NIFTY (Tuesdays).
        
        Args: num_weeks – number of weeks to look ahead (default 4).
        Returns: List of upcoming Tuesday expiry dates.
        Raises: None.
        """
        today = date.today()
        expiries = []
        
        # Find next Tuesday
        days_until_tuesday = (1 - today.weekday()) % 7
        if days_until_tuesday == 0:
            days_until_tuesday = 7  # If today is Tuesday, get next week
        
        next_tuesday = today + timedelta(days=days_until_tuesday)
        
        # Get num_weeks Tuesdays
        for i in range(num_weeks):
            expiry = next_tuesday + timedelta(weeks=i)
            expiries.append(expiry)
        
        return expiries

    def _parse_expiry(self, expiry_raw: Any) -> Optional[date]:
        """Parse expiry from broker data (may be date, datetime, or string).

        Args: expiry_raw – raw expiry value from broker instrument dump.
        Returns: date or None if unparseable.
        Raises: None.
        """
        if expiry_raw is None:
            return None
        if isinstance(expiry_raw, date) and not isinstance(expiry_raw, datetime):
            return expiry_raw
        if isinstance(expiry_raw, datetime):
            return expiry_raw.date()
        expiry_str = str(expiry_raw).strip()
        if not expiry_str:
            return None
        for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d-%b-%Y", "%d-%m-%Y"):
            try:
                return datetime.strptime(expiry_str, fmt).date()
            except ValueError:
                continue
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def resolve_by_symbol(self, symbol: str) -> Optional[dict]:
        """Resolve full instrument data by symbol (compatibility shim for OrderManager).

        Args: symbol – bare or exchange-qualified tradingsymbol.
        Returns: instrument dict or None if not found.
        Raises: None.
        """
        return self.lookup(symbol)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _symbol_by_token_set(
        self, token: int, tradingsymbol: str, exchange: str
    ) -> None:
        """Populate reverse maps. Must be called under self._lock."""
        self._symbol_map[token] = tradingsymbol
        self._exchange_map[token] = exchange

    # ------------------------------------------------------------------
    # Helpers replacing InstrumentResolver functionality
    # ------------------------------------------------------------------

    def get_option_contracts(self, underlying: str) -> List[Dict[str, Any]]:
        """Return option contracts for an underlying (e.g., 'NIFTY').

        Uses broker-provided structured data (expiry, strike, instrument_type)
        from the instrument dump instead of parsing symbol strings.

        Args: underlying – base name like 'NIFTY' or 'BANKNIFTY'.
        Returns: List of contract dicts with keys: instrument_token, tradingsymbol,
                 expiry, strike, instrument_type, lot_size.
        Raises: None.
        """
        key = str(underlying).strip().upper()
        today = date.today()
        contracts: List[Dict[str, Any]] = []

        with self._lock:
            for token, inst_data in self._instrument_data.items():
                inst_type = str(inst_data.get("instrument_type", "")).strip().upper()
                if inst_type not in ("CE", "PE"):
                    continue

                # Verify the underlying name matches
                name = str(inst_data.get("name", "")).strip().upper()
                if name != key:
                    continue

                # Parse expiry from broker data
                expiry_date = self._parse_expiry(inst_data.get("expiry"))
                if expiry_date is None or expiry_date < today:
                    continue

                # Parse strike from broker data
                try:
                    strike = float(inst_data.get("strike", 0))
                except (TypeError, ValueError):
                    continue
                if strike <= 0:
                    continue

                tradingsymbol = str(
                    inst_data.get("tradingsymbol", self._symbol_map.get(token, ""))
                )
                actual_lot_size = self._lot_size_map.get(token, 65)
                contracts.append({
                    "instrument_token": token,
                    "tradingsymbol": tradingsymbol,
                    "expiry": expiry_date,
                    "strike": strike,
                    "instrument_type": inst_type,
                    "lot_size": actual_lot_size,
                })

        return sorted(contracts, key=lambda c: (c["expiry"], c["strike"]))

    def get_future_contracts(self, underlying: str) -> List[Dict[str, Any]]:
        """Return futures contracts for an underlying (e.g., 'NIFTY').

        Uses broker-provided structured data from the instrument dump.

        Args: underlying – base name like 'NIFTY' or 'BANKNIFTY'.
        Returns: List of contract dicts with keys: instrument_token, tradingsymbol,
                 expiry, instrument_type="FUT", lot_size.
        Raises: None.
        """
        key = str(underlying).strip().upper()
        today = date.today()
        contracts: List[Dict[str, Any]] = []

        with self._lock:
            for token, inst_data in self._instrument_data.items():
                inst_type = str(inst_data.get("instrument_type", "")).strip().upper()
                if inst_type != "FUT":
                    continue

                name = str(inst_data.get("name", "")).strip().upper()
                if name != key:
                    continue

                expiry_date = self._parse_expiry(inst_data.get("expiry"))
                if expiry_date is None or expiry_date < today:
                    continue

                tradingsymbol = str(
                    inst_data.get("tradingsymbol", self._symbol_map.get(token, ""))
                )
                actual_lot_size = self._lot_size_map.get(token, 65)
                contracts.append({
                    "instrument_token": token,
                    "tradingsymbol": tradingsymbol,
                    "expiry": expiry_date,
                    "instrument_type": "FUT",
                    "lot_size": actual_lot_size,
                })

        return sorted(contracts, key=lambda c: c["expiry"])

    def select_tokens_for_universe(
        self,
        base: str = "NIFTY",
        spot_price: float | None = None,
        strikes_around_atm: int = 2,
        strike_step: int = 50,
    ) -> list[int]:
        """Select tokens for the trading universe including spot, futures, and ATM options.

        Uses broker-provided structured data from the instrument dump.

        Args:
            base – underlying name ('NIFTY', 'BANKNIFTY').
            spot_price – current spot price for ATM calculation.
            strikes_around_atm – number of strikes around ATM to include.
            strike_step – strike interval (default 50 for NIFTY).

        Returns: Sorted list of instrument tokens.
        """
        tokens: set[int] = set()
        base_upper = str(base).strip().upper()

        # 1. Spot token
        if base_upper in _WELL_KNOWN_TOKENS:
            tokens.add(_WELL_KNOWN_TOKENS[base_upper])

        # 2. Nearest future
        today = date.today()
        nearest_future_token: Optional[int] = None
        nearest_future_expiry: Optional[date] = None

        with self._lock:
            for token, inst_data in self._instrument_data.items():
                inst_type = str(inst_data.get("instrument_type", "")).strip().upper()
                name = str(inst_data.get("name", "")).strip().upper()
                if inst_type != "FUT" or name != base_upper:
                    continue

                expiry_date = self._parse_expiry(inst_data.get("expiry"))
                if expiry_date is None or expiry_date < today:
                    continue

                if nearest_future_expiry is None or expiry_date < nearest_future_expiry:
                    nearest_future_expiry = expiry_date
                    nearest_future_token = token

        if nearest_future_token is not None:
            tokens.add(nearest_future_token)

        # 3. ATM options for nearest expiry
        if spot_price and spot_price > 0 and nearest_future_expiry:
            atm_strike = _atm_strike_for_spot(spot_price, strike_step)
            actual_around = max(2, strikes_around_atm)
            target_strikes = {
                atm_strike + (i * strike_step)
                for i in range(-actual_around, actual_around + 1)
            }

            with self._lock:
                for token, inst_data in self._instrument_data.items():
                    inst_type = str(inst_data.get("instrument_type", "")).strip().upper()
                    name = str(inst_data.get("name", "")).strip().upper()
                    if inst_type not in ("CE", "PE") or name != base_upper:
                        continue

                    expiry_date = self._parse_expiry(inst_data.get("expiry"))
                    if expiry_date != nearest_future_expiry:
                        continue

                    try:
                        strike = float(inst_data.get("strike", 0))
                    except (TypeError, ValueError):
                        continue

                    if strike in target_strikes:
                        tokens.add(token)

        return sorted(list(tokens))

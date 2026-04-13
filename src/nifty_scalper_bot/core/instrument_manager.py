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
from datetime import date, datetime
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
                expiry_str = inst_data.get("expiry")
                if expiry_str:
                    try:
                        # Parse expiry string (format: YYYY-MM-DD)
                        inst_expiry = datetime.strptime(expiry_str, "%Y-%m-%d").date()
                        if inst_expiry == expiry_date:
                            result.append(dict(inst_data))
                    except (ValueError, TypeError):
                        pass
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
    # Additional helpers to replace InstrumentResolver functionality
    # ------------------------------------------------------------------

    def get_option_contracts(self, underlying: str) -> List[Dict[str, Any]]:
        """Return option contracts for an underlying (e.g., 'NIFTY').
        
        Args: underlying – base name like 'NIFTY' or 'BANKNIFTY'.
        Returns: List of contract dicts with keys: instrument_token, tradingsymbol, 
                 expiry, strike, instrument_type, lot_size.
        Raises: None.
        """
        key = str(underlying).strip().upper()
        today = date.today()
        contracts: List[Dict[str, Any]] = []
        
        with self._lock:
            for token, symbol in self._symbol_map.items():
                sym_upper = symbol.upper()
                if not sym_upper.startswith(key):
                    continue
                if not any(sym_upper.endswith(x) for x in ("CE", "PE")):
                    continue
                
                # Extract expiry and strike from symbol
                # Format: NIFTY26APR25600CE or NIFTY2641325600CE (weekly)
                rest = sym_upper[len(key):]
                
                # Try to parse expiry and strike
                expiry_date: Optional[date] = None
                strike: float = 0.0
                inst_type = "CE" if sym_upper.endswith("CE") else "PE"
                
                # Remove CE/PE suffix
                core = rest[:-2]
                
                # Parse based on format
                if len(core) >= 7:  # Monthly: 26APR25600
                    # Find where digits start after month code
                    idx = 0
                    while idx < len(core) and not core[idx].isdigit():
                        idx += 1
                    if idx >= 3:
                        expiry_str = core[:idx]
                        strike_str = core[idx:]
                        try:
                            # Parse YYMMM format
                            yy = int(expiry_str[:2])
                            month_str = expiry_str[2:].upper()
                            months = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
                                     "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}
                            month = months.get(month_str, 0)
                            if month:
                                year = 2000 + yy if yy < 50 else 1900 + yy
                                # Get last Tuesday of month
                                import calendar
                                last_day = calendar.monthrange(year, month)[1]
                                exp_dt = datetime(year, month, last_day)
                                while exp_dt.weekday() != 1:  # Tuesday
                                    exp_dt = exp_dt.replace(day=exp_dt.day - 1)
                                expiry_date = exp_dt.date()
                            strike = float(strike_str)
                        except (ValueError, IndexError):
                            pass
                elif len(core) >= 6:  # Weekly: 2641325600
                    try:
                        yy = int(core[:2])
                        mm = int(core[2:3]) if core[2].isdigit() else 0
                        dd = int(core[3:5])
                        strike_str = core[5:]
                        year = 2000 + yy if yy < 50 else 1900 + yy
                        # Map single digit month codes (O=10, N=11, D=12)
                        if mm == 0 and len(core) > 2:
                            mc = core[2]
                            mm_map = {"O": 10, "N": 11, "D": 12}
                            mm = mm_map.get(mc, 0)
                        if mm and dd:
                            expiry_date = date(year, mm, dd)
                        strike = float(strike_str)
                    except (ValueError, IndexError):
                        pass
                
                if expiry_date and expiry_date >= today and strike > 0:
                    # Get actual lot size from broker data - NIFTY is now 65
                    actual_lot_size = self._lot_size_map.get(token, 65)
                    contracts.append({
                        "instrument_token": token,
                        "tradingsymbol": symbol,
                        "expiry": expiry_date,
                        "strike": strike,
                        "instrument_type": inst_type,
                        "lot_size": actual_lot_size,
                    })
        
        return sorted(contracts, key=lambda c: (c["expiry"], c["strike"]))

    def get_future_contracts(self, underlying: str) -> List[Dict[str, Any]]:
        """Return futures contracts for an underlying (e.g., 'NIFTY').
        
        Args: underlying – base name like 'NIFTY' or 'BANKNIFTY'.
        Returns: List of contract dicts with keys: instrument_token, tradingsymbol, 
                 expiry, instrument_type="FUT", lot_size.
        Raises: None.
        """
        key = str(underlying).strip().upper()
        today = date.today()
        contracts: List[Dict[str, Any]] = []
        
        with self._lock:
            for token, symbol in self._symbol_map.items():
                sym_upper = symbol.upper()
                if not sym_upper.startswith(key):
                    continue
                if not sym_upper.endswith("FUT"):
                    continue
                
                # Extract expiry from future symbol
                rest = sym_upper[len(key):-3]  # Remove FUT suffix
                expiry_date: Optional[date] = None
                
                if len(rest) >= 5:  # 26APR format
                    try:
                        yy = int(rest[:2])
                        month_str = rest[2:5].upper()
                        months = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
                                 "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}
                        month = months.get(month_str, 0)
                        if month:
                            year = 2000 + yy if yy < 50 else 1900 + yy
                            import calendar
                            last_day = calendar.monthrange(year, month)[1]
                            exp_dt = datetime(year, month, last_day)
                            while exp_dt.weekday() != 1:  # Tuesday
                                exp_dt = exp_dt.replace(day=exp_dt.day - 1)
                            expiry_date = exp_dt.date()
                    except (ValueError, IndexError):
                        pass
                
                if expiry_date and expiry_date >= today:
                    # Get actual lot size from broker data - NIFTY is now 65
                    actual_lot_size = self._lot_size_map.get(token, 65)
                    contracts.append({
                        "instrument_token": token,
                        "tradingsymbol": symbol,
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
        spot_key = base_upper
        if spot_key in _WELL_KNOWN_TOKENS:
            tokens.add(_WELL_KNOWN_TOKENS[spot_key])
        
        # 2. Nearest future
        today = date.today()
        nearest_future: Optional[Dict[str, Any]] = None
        nearest_future_expiry: Optional[date] = None
        
        with self._lock:
            for token, symbol in self._symbol_map.items():
                sym_upper = symbol.upper()
                if not sym_upper.startswith(base_upper):
                    continue
                if not sym_upper.endswith("FUT"):
                    continue
                
                # Parse expiry from future symbol
                rest = sym_upper[len(base_upper):-3]  # Remove FUT suffix
                expiry_date: Optional[date] = None
                
                if len(rest) >= 5:  # 26APR format
                    try:
                        yy = int(rest[:2])
                        month_str = rest[2:5].upper()
                        months = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
                                 "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}
                        month = months.get(month_str, 0)
                        if month:
                            year = 2000 + yy if yy < 50 else 1900 + yy
                            import calendar
                            last_day = calendar.monthrange(year, month)[1]
                            exp_dt = datetime(year, month, last_day)
                            while exp_dt.weekday() != 1:
                                exp_dt = exp_dt.replace(day=exp_dt.day - 1)
                            expiry_date = exp_dt.date()
                    except (ValueError, IndexError):
                        pass
                
                if expiry_date and expiry_date >= today:
                    if nearest_future_expiry is None or expiry_date < nearest_future_expiry:
                        nearest_future_expiry = expiry_date
                        nearest_future = {"token": token, "symbol": symbol, "expiry": expiry_date}
        
        if nearest_future:
            tokens.add(nearest_future["token"])
        
        # 3. ATM options for nearest expiry
        if spot_price and spot_price > 0 and nearest_future_expiry:
            atm_strike = _atm_strike_for_spot(spot_price, strike_step)
            actual_around = max(2, strikes_around_atm)
            target_strikes = {
                atm_strike + (i * strike_step)
                for i in range(-actual_around, actual_around + 1)
            }
            
            with self._lock:
                for token, symbol in self._symbol_map.items():
                    sym_upper = symbol.upper()
                    if not sym_upper.startswith(base_upper):
                        continue
                    if not any(sym_upper.endswith(x) for x in ("CE", "PE")):
                        continue
                    
                    # Parse expiry and strike
                    rest = sym_upper[len(base_upper):]
                    expiry_date: Optional[date] = None
                    strike: float = 0.0
                    
                    ce_pe = "CE" if sym_upper.endswith("CE") else "PE"
                    core = rest[:-2]
                    
                    if len(core) >= 7:  # Monthly
                        idx = 0
                        while idx < len(core) and not core[idx].isdigit():
                            idx += 1
                        if idx >= 3:
                            expiry_str = core[:idx]
                            strike_str = core[idx:]
                            try:
                                yy = int(expiry_str[:2])
                                month_str = expiry_str[2:].upper()
                                months = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
                                         "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}
                                month = months.get(month_str, 0)
                                if month:
                                    year = 2000 + yy if yy < 50 else 1900 + yy
                                    import calendar
                                    last_day = calendar.monthrange(year, month)[1]
                                    exp_dt = datetime(year, month, last_day)
                                    while exp_dt.weekday() != 1:
                                        exp_dt = exp_dt.replace(day=exp_dt.day - 1)
                                    expiry_date = exp_dt.date()
                                strike = float(strike_str)
                            except (ValueError, IndexError):
                                pass
                    elif len(core) >= 6:  # Weekly
                        try:
                            yy = int(core[:2])
                            mc = core[2]
                            dd = int(core[3:5])
                            strike_str = core[5:]
                            year = 2000 + yy if yy < 50 else 1900 + yy
                            mm_map = {"O": 10, "N": 11, "D": 12}
                            mm = mm_map.get(mc, 0)
                            if mm and dd:
                                expiry_date = date(year, mm, dd)
                            strike = float(strike_str)
                        except (ValueError, IndexError):
                            pass
                    
                    if (expiry_date and expiry_date == nearest_future_expiry 
                        and strike in target_strikes):
                        tokens.add(token)
        
        return sorted(list(tokens))

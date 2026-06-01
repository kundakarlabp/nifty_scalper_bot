"""Single source of truth for instrument token/symbol mappings.

Runtime role:
- SSOT for NIFTY instrument identity, symbol-token mapping, and active contract selection.
- Owns NIFTY spot token, active future, selected ATM CE/PE, nearby option universe.
- Other modules must consume ActiveContractBasket and must not regenerate symbols.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Mapping, Optional

LOGGER = logging.getLogger("nifty_scalper_bot.core.instrument_manager")

# Well-known index tokens
_WELL_KNOWN_TOKENS: Dict[str, int] = {
    "NIFTY": 256265,
    "BANKNIFTY": 260105,
    "NSE:NIFTY": 256265,
    "NSE:BANKNIFTY": 260105,
}


@dataclass(frozen=True, slots=True)
class ActiveContractBasket:
    spot_symbol: str
    spot_token: int

    futures_symbol: str | None
    futures_token: int | None
    futures_expiry: date | None

    selected_ce: str
    selected_ce_token: int
    selected_pe: str
    selected_pe_token: int

    option_expiry: date
    atm_strike: int

    option_symbols: tuple[str, ...]
    option_tokens: tuple[int, ...]

    all_symbols: tuple[str, ...]
    all_tokens: tuple[int, ...]

    token_by_symbol: Mapping[str, int]
    symbol_by_token: Mapping[int, str]

    metadata: Mapping[str, Any]


def _exchange_trading_date() -> date:
    """Return the current NSE trading date in IST. TODO: holiday calendar handling."""
    return datetime.now(ZoneInfo("Asia/Kolkata")).date()


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

    def get_nifty_instruments_snapshot(self) -> List[Dict[str, Any]]:
        """Return cached NIFTY instrument rows. Args: none. Returns: rows. Raises: none."""

        with self._lock:
            return [dict(row) for row in self._instrument_data.values()]

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

    def lot_size_for_symbol(self, symbol: str) -> Optional[int]:
        """Compatibility alias for lot-size resolver consumers.

        Args: symbol – exchange-qualified or bare trading symbol.
        Returns: Lot size or ``None`` when not available.
        Raises: None.
        """

        return self.get_lot_size(symbol)

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
        today = _exchange_trading_date()
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
        today = _exchange_trading_date()
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
        today = _exchange_trading_date()
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

    @staticmethod
    def _exchange_qualified(row: Mapping[str, Any]) -> str:
        exchange = str(row.get("exchange") or "NFO").strip().upper()
        tradingsymbol = str(row.get("tradingsymbol") or "").strip().upper()
        return f"{exchange}:{tradingsymbol}"

    def _nifty_rows_snapshot(self) -> list[tuple[int, dict[str, Any]]]:
        if not self.is_loaded():
            self.load()
        with self._lock:
            rows = [(int(token), dict(row)) for token, row in self._instrument_data.items()]
        if not rows:
            raise RuntimeError("CONTRACT_SSOT_INSTRUMENTS_MISSING reason=no_nfo_instruments")
        LOGGER.info(
            "CONTRACT_SSOT_INSTRUMENTS_READY count=%d",
            len(rows),
            extra={"event": "CONTRACT_SSOT_INSTRUMENTS_READY", "count": len(rows)},
        )
        return rows

    def get_active_nifty_contracts(
        self,
        spot_price: float,
        *,
        strikes_around_atm: int = 2,
        strike_step: int = 50,
        include_future: bool = True,
    ) -> ActiveContractBasket:
        """Resolve the authoritative live NIFTY spot/future/option basket."""
        try:
            spot = float(spot_price)
        except (TypeError, ValueError) as exc:
            raise ValueError("spot_price must be positive") from exc
        if spot <= 0:
            raise ValueError("spot_price must be positive")
        if strike_step <= 0:
            raise ValueError("strike_step must be positive")

        today = _exchange_trading_date()
        rows = self._nifty_rows_snapshot()

        futures: list[tuple[date, int, dict[str, Any]]] = []
        options: list[tuple[date, int, float, str, dict[str, Any]]] = []
        for token, row in rows:
            name = str(row.get("name") or "").strip().upper()
            if name != "NIFTY":
                continue
            inst_type = str(row.get("instrument_type") or "").strip().upper()
            expiry = self._parse_expiry(row.get("expiry"))
            if expiry is None or expiry < today:
                continue
            if inst_type == "FUT":
                futures.append((expiry, token, row))
                continue
            if inst_type in {"CE", "PE"}:
                try:
                    strike = float(row.get("strike"))
                except (TypeError, ValueError):
                    continue
                if strike <= 0:
                    continue
                options.append((expiry, token, strike, inst_type, row))

        futures.sort(key=lambda item: (item[0], self._exchange_qualified(item[2]), item[1]))
        future_symbol: str | None = None
        future_token: int | None = None
        future_expiry: date | None = None
        if include_future and futures:
            future_expiry, future_token, future_row = futures[0]
            future_symbol = self._exchange_qualified(future_row)
            LOGGER.info(
                "CONTRACT_SSOT_FUTURE_SELECTED symbol=%s token=%s expiry=%s",
                future_symbol, future_token, future_expiry,
                extra={"event": "CONTRACT_SSOT_FUTURE_SELECTED", "symbol": future_symbol, "token": future_token, "expiry": str(future_expiry)},
            )
        elif include_future:
            LOGGER.warning(
                "CONTRACT_SSOT_FUTURE_SELECTED symbol=None reason=active_future_unresolved",
                extra={"event": "CONTRACT_SSOT_FUTURE_SELECTED", "reason": "active_future_unresolved"},
            )

        if not options:
            raise RuntimeError("CONTRACT_SSOT_OPTIONS_MISSING reason=no_valid_nifty_options")
        option_expiry = min(item[0] for item in options)
        LOGGER.info(
            "CONTRACT_SSOT_OPTION_EXPIRY_SELECTED expiry=%s",
            option_expiry,
            extra={"event": "CONTRACT_SSOT_OPTION_EXPIRY_SELECTED", "expiry": str(option_expiry)},
        )

        atm = _atm_strike_for_spot(spot, int(strike_step))
        around = max(0, int(strikes_around_atm))
        target_strikes = {float(atm + i * int(strike_step)) for i in range(-around, around + 1)}
        expiry_options = [item for item in options if item[0] == option_expiry]

        selected_ce = next((item for item in expiry_options if item[2] == float(atm) and item[3] == "CE"), None)
        selected_pe = next((item for item in expiry_options if item[2] == float(atm) and item[3] == "PE"), None)
        if selected_ce is None or selected_pe is None:
            raise RuntimeError(
                f"CONTRACT_SSOT_ATM_PAIR_MISSING atm_strike={atm} expiry={option_expiry} "
                f"ce_found={selected_ce is not None} pe_found={selected_pe is not None}"
            )

        def opt_sort(item: tuple[date, int, float, str, dict[str, Any]]) -> tuple[float, int, str]:
            side_rank = 0 if item[3] == "CE" else 1
            return (item[2], side_rank, self._exchange_qualified(item[4]))

        selected_options = sorted(
            [item for item in expiry_options if item[2] in target_strikes],
            key=opt_sort,
        )
        required_tokens = {selected_ce[1], selected_pe[1]}
        if not required_tokens.issubset({item[1] for item in selected_options}):
            selected_options.extend(item for item in (selected_ce, selected_pe) if item[1] not in {x[1] for x in selected_options})
            selected_options = sorted(selected_options, key=opt_sort)

        option_symbols = tuple(self._exchange_qualified(item[4]) for item in selected_options)
        option_tokens = tuple(int(item[1]) for item in selected_options)
        ce_symbol = self._exchange_qualified(selected_ce[4])
        pe_symbol = self._exchange_qualified(selected_pe[4])
        LOGGER.info(
            "CONTRACT_SSOT_ATM_PAIR_SELECTED selected_ce=%s selected_pe=%s atm_strike=%s",
            ce_symbol, pe_symbol, atm,
            extra={"event": "CONTRACT_SSOT_ATM_PAIR_SELECTED", "selected_ce": ce_symbol, "selected_pe": pe_symbol, "atm_strike": atm},
        )

        all_symbols_list = ["NSE:NIFTY"]
        all_tokens_list = [int(_WELL_KNOWN_TOKENS["NSE:NIFTY"])]
        if future_symbol and future_token is not None:
            all_symbols_list.append(future_symbol)
            all_tokens_list.append(int(future_token))
        for sym, tok in zip(option_symbols, option_tokens):
            if tok not in all_tokens_list:
                all_symbols_list.append(sym)
                all_tokens_list.append(tok)
        token_by_symbol = dict(zip(all_symbols_list, all_tokens_list))
        symbol_by_token = {token: symbol for symbol, token in token_by_symbol.items()}
        basket = ActiveContractBasket(
            spot_symbol="NSE:NIFTY",
            spot_token=int(_WELL_KNOWN_TOKENS["NSE:NIFTY"]),
            futures_symbol=future_symbol,
            futures_token=future_token,
            futures_expiry=future_expiry,
            selected_ce=ce_symbol,
            selected_ce_token=int(selected_ce[1]),
            selected_pe=pe_symbol,
            selected_pe_token=int(selected_pe[1]),
            option_expiry=option_expiry,
            atm_strike=atm,
            option_symbols=option_symbols,
            option_tokens=option_tokens,
            all_symbols=tuple(all_symbols_list),
            all_tokens=tuple(all_tokens_list),
            token_by_symbol=token_by_symbol,
            symbol_by_token=symbol_by_token,
            metadata={
                "source": "instrument_manager",
                "spot_price": spot,
                "strikes_around_atm": around,
                "strike_step": int(strike_step),
                "future_available": bool(future_symbol),
            },
        )
        LOGGER.info(
            "CONTRACT_SSOT_BASKET_SELECTED selected_ce=%s selected_pe=%s futures_symbol=%s atm_strike=%s option_count=%d token_count=%d",
            basket.selected_ce, basket.selected_pe, basket.futures_symbol, basket.atm_strike, len(basket.option_symbols), len(basket.all_tokens),
            extra={"event": "CONTRACT_SSOT_BASKET_SELECTED", "selected_ce": basket.selected_ce, "selected_pe": basket.selected_pe, "futures_symbol": basket.futures_symbol, "atm_strike": basket.atm_strike, "option_count": len(basket.option_symbols), "token_count": len(basket.all_tokens)},
        )
        return basket

    def select_tokens_for_universe(
        self,
        base: str = "NIFTY",
        spot_price: float | None = None,
        strikes_around_atm: int = 2,
        strike_step: int = 50,
    ) -> list[int]:
        """Select tokens using independent future and option expiries from InstrumentManager SSOT."""
        base_upper = str(base).strip().upper()
        if base_upper == "NIFTY" and spot_price and spot_price > 0:
            basket = self.get_active_nifty_contracts(
                float(spot_price),
                strikes_around_atm=strikes_around_atm,
                strike_step=strike_step,
                include_future=True,
            )
            return list(basket.all_tokens)

        tokens: set[int] = set()
        if base_upper in _WELL_KNOWN_TOKENS:
            tokens.add(_WELL_KNOWN_TOKENS[base_upper])

        today = _exchange_trading_date()
        nearest_future: tuple[date, int] | None = None
        nearest_option_expiry: date | None = None
        with self._lock:
            for token, inst_data in self._instrument_data.items():
                name = str(inst_data.get("name", "")).strip().upper()
                if name != base_upper:
                    continue
                inst_type = str(inst_data.get("instrument_type", "")).strip().upper()
                expiry_date = self._parse_expiry(inst_data.get("expiry"))
                if expiry_date is None or expiry_date < today:
                    continue
                if inst_type == "FUT":
                    if nearest_future is None or expiry_date < nearest_future[0]:
                        nearest_future = (expiry_date, int(token))
                elif inst_type in {"CE", "PE"}:
                    if nearest_option_expiry is None or expiry_date < nearest_option_expiry:
                        nearest_option_expiry = expiry_date
        if nearest_future is not None:
            tokens.add(nearest_future[1])

        if spot_price and spot_price > 0 and nearest_option_expiry is not None:
            atm_strike = _atm_strike_for_spot(float(spot_price), strike_step)
            actual_around = max(2, strikes_around_atm)
            target_strikes = {atm_strike + (i * strike_step) for i in range(-actual_around, actual_around + 1)}
            with self._lock:
                for token, inst_data in self._instrument_data.items():
                    name = str(inst_data.get("name", "")).strip().upper()
                    inst_type = str(inst_data.get("instrument_type", "")).strip().upper()
                    if name != base_upper or inst_type not in {"CE", "PE"}:
                        continue
                    expiry_date = self._parse_expiry(inst_data.get("expiry"))
                    if expiry_date != nearest_option_expiry:
                        continue
                    try:
                        strike = float(inst_data.get("strike", 0))
                    except (TypeError, ValueError):
                        continue
                    if strike in target_strikes:
                        tokens.add(int(token))
        return sorted(tokens)

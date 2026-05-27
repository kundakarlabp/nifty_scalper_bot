"""Instrument lookup helpers for Kite instruments."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

import pandas as pd


@dataclass
class Instrument:
    exchange: str
    tradingsymbol: str
    instrument_token: int
    name: Optional[str] = None
    expiry: Optional[str] = None  # YYYY-MM-DD
    strike: Optional[float] = None
    option_type: Optional[str] = None  # 'CE' or 'PE' or None


class InstrumentLookup:
    """
    Robust lookup for Kite instrument tokens.

    Usage:
        il = InstrumentLookup('path/to/instruments.csv')
        token = il.get_token(exchange='NFO', tradingsymbol='NIFTY24DEC17400CE')
        token = il.get_token_by_fields(
            exchange='NFO', underlying='NIFTY', expiry='2024-12-19',
            strike=17400, option_type='CE'
        )
    """

    def __init__(self, csv_path: str | PathLike[str] | None = None, ttl_seconds: int = 600):
        self.csv_path = csv_path
        self.ttl_seconds = int(ttl_seconds)
        self._lock = threading.RLock()
        self._loaded_at = 0.0
        self._by_exchange_symbol: Dict[Tuple[str, str], Instrument] = {}
        self._by_option_fields: Dict[Tuple[str, str, str, float, str], Instrument] = {}
        self._warned_missing: set[str] = set()
        try:
            path_candidate = Path(csv_path) if csv_path is not None else None
        except (TypeError, ValueError):
            path_candidate = None
        if path_candidate is not None and path_candidate.exists():
            self.csv_path = str(path_candidate)
            self.reload()
        elif path_candidate is not None and type(csv_path).__module__ == "builtins":
            raise FileNotFoundError(f"Instrument CSV not found: {path_candidate}")
        else:
            self.csv_path = None
            self._loaded_at = time.time()

    def _normalize_expiry(self, expiry) -> Optional[str]:
        if expiry is None:
            return None
        if isinstance(expiry, str) and len(expiry) == 10 and expiry[4] == "-":
            return expiry
        try:
            dt = pd.to_datetime(expiry, utc=True).date()
            return dt.isoformat()
        except Exception:
            return None

    def reload(self) -> None:
        """Args: None. Returns: None. Raises: RuntimeError."""
        with self._lock:
            df = pd.read_csv(self.csv_path, dtype=str, low_memory=False)
            expected = ["exchange", "tradingsymbol", "instrument_token"]
            for col in expected:
                if col not in (c.lower() for c in df.columns):
                    raise RuntimeError(
                        f"instruments CSV missing required column: {col}"
                    )

            self._by_exchange_symbol.clear()
            self._by_option_fields.clear()

            for _, row in df.iterrows():
                exch = str(row["exchange"]).strip()
                sym = str(row["tradingsymbol"]).strip()
                token = int(row["instrument_token"])
                name = row.get("name") if "name" in row else None
                expiry = None
                strike = None
                opt_type = None

                rlower = {k.lower(): v for k, v in row.items()}
                for col in ("expiry", "expiry_date", "expiry_dt"):
                    if col in rlower and pd.notna(rlower[col]):
                        expiry = self._normalize_expiry(rlower[col])
                        break
                for col in ("strike", "strike_price"):
                    if col in rlower and pd.notna(rlower[col]):
                        try:
                            strike = float(rlower[col])
                        except Exception:
                            strike = None
                        break
                for col in ("instrument_type", "option_type", "segment"):
                    if col in rlower and pd.notna(rlower[col]):
                        val = str(rlower[col]).upper()
                        if val in ("CE", "PE"):
                            opt_type = val
                            break

                inst = Instrument(
                    exchange=exch,
                    tradingsymbol=sym,
                    instrument_token=token,
                    name=name,
                    expiry=expiry,
                    strike=strike,
                    option_type=opt_type,
                )

                key = (exch.upper(), sym.upper())
                self._by_exchange_symbol[key] = inst

                if opt_type and expiry and strike is not None:
                    under = sym.upper()
                    base = ""
                    for ch in under:
                        if ch.isalpha():
                            base += ch
                        else:
                            break
                    base = base or under
                    opt_key = (
                        exch.upper(),
                        base,
                        expiry,
                        float(strike),
                        opt_type.upper(),
                    )
                    self._by_option_fields[opt_key] = inst

            self._loaded_at = time.time()

    def _ensure_fresh(self) -> None:
        if isinstance(self.csv_path, str) and time.time() - self._loaded_at > self.ttl_seconds:
            self.reload()

    def upsert(
        self,
        tradingsymbol: str,
        instrument_token: int,
        *,
        exchange: str = "NFO",
        name: str | None = None,
        expiry: str | None = None,
        strike: float | None = None,
        option_type: str | None = None,
    ) -> None:
        inst = Instrument(
            exchange=str(exchange).upper(),
            tradingsymbol=str(tradingsymbol).upper(),
            instrument_token=int(instrument_token),
            name=name,
            expiry=self._normalize_expiry(expiry),
            strike=float(strike) if strike is not None else None,
            option_type=str(option_type).upper() if option_type else None,
        )
        with self._lock:
            self._by_exchange_symbol[(inst.exchange, inst.tradingsymbol)] = inst
            if inst.option_type and inst.expiry and inst.strike is not None:
                under = inst.tradingsymbol
                base = ""
                for ch in under:
                    if ch.isalpha():
                        base += ch
                    else:
                        break
                base = base or under
                self._by_option_fields[
                    (inst.exchange, base, inst.expiry, float(inst.strike), inst.option_type)
                ] = inst

    def resolve(self, symbol: str, exchange: str = "NFO") -> int | None:
        try:
            return self.get_token(exchange, symbol)
        except KeyError:
            key = str(symbol).upper()
            logger = logging.getLogger("nifty_scalper_bot.data.instruments")
            if key not in self._warned_missing:
                self._warned_missing.add(key)
                logger.warning("resolver no token for %s", symbol)
            else:
                logger.debug("resolver no token warning suppressed for %s", symbol)
            return None

    def lookup(self, symbol: str, exchange: str = "NFO") -> dict[str, object]:
        token = self.resolve(symbol, exchange=exchange)
        return {"symbol": str(symbol), "exchange": str(exchange).upper(), "token": token}

    def get_token(self, exchange: str, tradingsymbol: str) -> int:
        """Args: exchange, tradingsymbol. Returns: token. Raises: KeyError."""
        with self._lock:
            self._ensure_fresh()
            key = (exchange.upper(), tradingsymbol.upper())
            inst = self._by_exchange_symbol.get(key)
            if inst is None:
                raise KeyError(
                    f"No instrument for exchange={exchange} "
                    f"tradingsymbol={tradingsymbol}"
                )
            return inst.instrument_token

    def get_token_by_fields(
        self,
        exchange: str,
        underlying: str,
        expiry: str,
        strike: float,
        option_type: str,
    ) -> int:
        """Args: exch, under, expiry, strike, opt; Returns: token; Raises: Key."""
        with self._lock:
            self._ensure_fresh()
            expiry_norm = self._normalize_expiry(expiry)
            if expiry_norm is None:
                raise KeyError(
                    f"No option instrument for {exchange} {underlying} "
                    f"{expiry} {strike} {option_type}"
                )
            key = (
                exchange.upper(),
                underlying.upper(),
                expiry_norm,
                float(strike),
                option_type.upper(),
            )
            inst = self._by_option_fields.get(key)
            if inst:
                return inst.instrument_token
            for k, v in self._by_option_fields.items():
                (exch, _base, exp, st, ot) = k
                if (
                    exch == exchange.upper()
                    and exp == expiry_norm
                    and st == float(strike)
                    and ot == option_type.upper()
                ):
                    return v.instrument_token
            raise KeyError(
                f"No option instrument for {exchange} {underlying} "
                f"{expiry_norm} {strike} {option_type}"
            )

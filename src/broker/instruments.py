"""Instrument models and store."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class Instrument:
    """Trading instrument details."""

    token: int
    symbol: str
    exchange: str = "NFO"
    product: str = "MIS"
    variety: str = "regular"
    lot_size: int = 1
    tick_size: float = 0.05


class InstrumentStore:
    """Lookup helper for instruments loaded from a CSV file."""

    def __init__(self, instruments: Iterable[Instrument] = ()): 
        self._by_token: Dict[int, Instrument] = {}
        self._by_symbol: Dict[str, Instrument] = {}
        for inst in instruments:
            self.add(inst)

    def add(self, inst: Instrument) -> None:
        """Add an instrument to the store."""
        self._by_token[int(inst.token)] = inst
        self._by_symbol[str(inst.symbol)] = inst

    @classmethod
    def from_csv(cls, path: str) -> "InstrumentStore":
        """Load instruments from a CSV file."""
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        items: List[Instrument] = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    items.append(
                        Instrument(
                            token=int(row.get("token") or row.get("instrument_token") or 0),
                            symbol=str(row.get("symbol") or row.get("tradingsymbol")),
                            exchange=str(row.get("exchange") or "NFO"),
                            product=str(row.get("product") or "MIS"),
                            variety=str(row.get("variety") or "regular"),
                            lot_size=int(row.get("lot_size") or 1),
                            tick_size=float(row.get("tick_size") or 0.05),
                        )
                    )
                except Exception:
                    continue
        return cls(items)

    def by_token(self, token: int) -> Optional[Instrument]:
        """Return instrument by token."""
        return self._by_token.get(int(token))

    def by_symbol(self, symbol: str) -> Optional[Instrument]:
        """Return instrument by symbol."""
        return self._by_symbol.get(str(symbol))

    def tradingsymbol(self, token: int) -> Optional[str]:
        """Return trading symbol for a token."""
        inst = self.by_token(token)
        return inst.symbol if inst else None

    def exchange(self, token: int) -> Optional[str]:
        """Return exchange name for a token."""
        inst = self.by_token(token)
        return inst.exchange if inst else None

    def product(self, token: int) -> Optional[str]:
        """Return product type for a token."""
        inst = self.by_token(token)
        return inst.product if inst else None

    def variety(self, token: int) -> Optional[str]:
        """Return order variety for a token."""
        inst = self.by_token(token)
        return inst.variety if inst else None

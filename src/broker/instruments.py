"""Instrument models and store."""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Instrument:
    """Trading instrument details."""

    token: int
    symbol: str
    exchange: str = "NFO"
    product: str = "MIS"
    variety: str = "regular"
    lot_size: int = 1
    tick_size: Decimal = Decimal("0.05")


class InstrumentStore:
    """Lookup helper for instruments loaded from a CSV file."""

    def __init__(self, instruments: Iterable[Instrument] = (), path: str | None = None):
        self._by_token: Dict[int, Instrument] = {}
        self._by_symbol: Dict[str, Instrument] = {}
        self._path = path
        for inst in instruments:
            self.add(inst)

    def add(self, inst: Instrument) -> None:
        """Add an instrument to the store."""
        self._by_token[int(inst.token)] = inst
        self._by_symbol[str(inst.symbol)] = inst

    @classmethod
    def from_csv(cls, path: str) -> "InstrumentStore":
        """Load instruments from a CSV file."""
        items = cls._read_csv(path)
        return cls(items, path=path)

    @staticmethod
    def _read_csv(path: str) -> List[Instrument]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        items: List[Instrument] = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                token_raw = row.get("token") or row.get("instrument_token")
                lot_raw = row.get("lot_size")
                tick_raw = row.get("tick_size")
                symbol = row.get("symbol") or row.get("tradingsymbol")
                if not (token_raw and lot_raw and tick_raw and symbol):
                    raise ValueError(f"Invalid row: {row}")
                try:
                    token = int(token_raw)
                    lot_size = int(lot_raw)
                    tick_size = Decimal(str(tick_raw))
                except Exception as exc:  # pragma: no cover - validation
                    raise ValueError(f"Invalid row: {row}") from exc
                items.append(
                    Instrument(
                        token=token,
                        symbol=str(symbol),
                        exchange=str(row.get("exchange") or "NFO"),
                        product=str(row.get("product") or "MIS"),
                        variety=str(row.get("variety") or "regular"),
                        lot_size=lot_size,
                        tick_size=tick_size,
                    )
                )
        return items

    def reload(self) -> None:
        """Reload instruments from the original CSV and log diffs."""
        if not self._path:
            raise ValueError("reload requires original CSV path")
        new_items = self._read_csv(self._path)
        new_by_token = {int(inst.token): inst for inst in new_items}
        added = [t for t in new_by_token if t not in self._by_token]
        removed = [t for t in self._by_token if t not in new_by_token]
        changed = [
            t
            for t in new_by_token
            if t in self._by_token and new_by_token[t] != self._by_token[t]
        ]
        if added or removed or changed:
            logger.info(
                "instrument reload diffs added=%s removed=%s changed=%s",
                added,
                removed,
                changed,
            )
        self._by_token = new_by_token
        self._by_symbol = {inst.symbol: inst for inst in new_items}

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

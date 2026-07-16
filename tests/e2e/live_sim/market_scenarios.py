from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class CEBreakoutScenario:
    spot_symbol: str = "NSE:NIFTY"
    future_symbol: str = "NFO:NIFTY26JULFUT"
    ce_symbol: str = "NFO:NIFTY26JUL25000CE"
    pe_symbol: str = "NFO:NIFTY26JUL25000PE"
    entry_price: float = 100.0
    target_price: float = 110.0
    initial_stop: float = 95.0
    lot_size: int = 75

    @property
    def phases(self) -> list[str]:
        return [
            "PREOPEN_HYDRATION",
            "MARKET_OPEN_STABILISATION",
            "CE_CONTEXT_BUILD",
            "CE_SIGNAL_TRIGGER",
            "ENTRY_FILL",
            "FAVOURABLE_MOVE",
            "BREAKEVEN_MOVE",
            "TRAILING_MOVE",
            "TARGET_OR_REVERSAL_EXIT",
            "POST_EXIT_RECONCILIATION",
        ]

    def live_ticks(self, start: datetime) -> list[tuple[str, float]]:
        return [
            (self.spot_symbol, 25000),
            (self.future_symbol, 25035),
            (self.ce_symbol, 99),
            (self.pe_symbol, 82),
        ]

"""Symbol/token resolution and strike selection for Nifty options."""

from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..broker.base import BrokerGateway


class InstrumentResolver:
    """Resolves option symbols, tokens, strikes, and expiries."""

    def __init__(self, broker: "BrokerGateway", underlying: str = "NIFTY") -> None:
        self._broker = broker
        self._underlying = underlying
        self._symbol_to_token: dict[str, int] = {}
        self._token_to_symbol: dict[int, str] = {}

    async def load_instruments(self, exchange: str = "NFO") -> None:
        instruments = await self._broker.get_instruments(exchange)
        for inst in instruments:
            sym = inst.get("tradingsymbol", "")
            token = inst.get("instrument_token", 0)
            if self._underlying in sym and sym.endswith(("CE", "PE")):
                self._symbol_to_token[sym] = token
                self._token_to_symbol[token] = sym

    def get_atm_strike(self, spot: float, interval: float = 50.0) -> float:
        return round(spot / interval) * interval

    def get_strike_range(self, spot: float, n: int = 5, interval: float = 50.0) -> list[float]:
        atm = self.get_atm_strike(spot, interval)
        return [atm + interval * i for i in range(-n, n + 1)]

    def get_token(self, symbol: str) -> int | None:
        return self._symbol_to_token.get(symbol)

    def get_symbol(self, token: int) -> str | None:
        return self._token_to_symbol.get(token)

    def get_weekly_expiry(self) -> date:
        today = date.today()
        days_ahead = (3 - today.weekday()) % 7  # Thursday=3
        if days_ahead == 0:
            days_ahead = 7
        return today + timedelta(days=days_ahead)

    def get_monthly_expiry(self) -> date:
        today = date.today()
        month = today.month + 1 if today.month < 12 else 1
        year = today.year if today.month < 12 else today.year + 1
        last = date(year, month + 1 if month < 12 else 1,
                    1 if month < 12 else 1) - timedelta(days=1)
        # Walk back to Thursday
        days_back = (last.weekday() - 3) % 7
        return last - timedelta(days=days_back)

    def resolve_option(self, underlying: str, strike: float, expiry: date, option_type: str) -> str:
        expiry_str = expiry.strftime("%y%b").upper()
        return f"{underlying}{expiry_str}{int(strike)}{option_type.upper()}"

    def get_subscribed_tokens(self) -> dict[int, str]:
        return dict(self._token_to_symbol)

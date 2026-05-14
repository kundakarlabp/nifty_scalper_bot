"""In-memory instrument resolver for symbol/token mapping."""

from __future__ import annotations

from typing import Any


class InstrumentResolver:
    """Resolve symbols to instrument metadata. Args: broker. Returns: resolver. Raises: none."""

    def __init__(self, broker: Any) -> None:
        self._broker = broker
        self._token_by_symbol: dict[str, int] = {}
        self._exchange_by_symbol: dict[str, str] = {}

    def warm_from_broker_dump(self, rows: list[dict[str, Any]]) -> None:
        """Warm caches from broker rows. Args: rows. Returns: none. Raises: none."""

        for row in rows:
            symbol = str(row.get('tradingsymbol') or '').strip().upper()
            if not symbol:
                continue
            token_raw = row.get('instrument_token')
            exchange = str(row.get('exchange') or 'NSE').strip().upper()
            try:
                token = int(token_raw)
            except (TypeError, ValueError):
                continue
            self._token_by_symbol[symbol] = token
            self._exchange_by_symbol[symbol] = exchange

    def resolve_symbol_to_token(self, symbol: str) -> int | None:
        """Resolve instrument token. Args: symbol. Returns: token/None. Raises: none."""

        canon, _, _ = self.canonicalize(symbol)
        return self._token_by_symbol.get(canon)

    def build_quote_keys(self, symbol: str) -> tuple[str, list[str]]:
        """Build quote lookup keys. Args: symbol. Returns: canonical+keys. Raises: none."""

        canon, exchange, _ = self.canonicalize(symbol)
        keys = [f'{exchange}:{canon}'] if exchange else []
        keys.append(canon)
        return canon, keys

    def canonicalize(self, symbol: str) -> tuple[str, str | None, str]:
        """Canonicalize symbol to routing tuple. Args: symbol. Returns: tuple. Raises: none."""

        raw = str(symbol or '').strip().upper()
        if ':' in raw:
            exchange, raw_symbol = raw.split(':', 1)
        else:
            exchange, raw_symbol = None, raw
        resolved_exchange = exchange or self._exchange_by_symbol.get(raw_symbol)
        if resolved_exchange in {'NFO', 'BFO'} or raw_symbol.endswith(('CE', 'PE')):
            return raw_symbol, resolved_exchange or 'NFO', 'OPTIONS'
        if raw_symbol in {'NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY'}:
            return raw_symbol, resolved_exchange or 'NSE', 'INDEX'
        return raw_symbol, resolved_exchange, 'UNKNOWN'

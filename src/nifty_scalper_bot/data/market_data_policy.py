"""Central policy for websocket-first market-data behavior."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse bool env var. Args: name/default. Returns: bool. Raises: none."""
    raw = str(os.getenv(name, str(default))).strip().lower()
    return raw in {'1', 'true', 'yes', 'y', 'on'}


def _env_float(name: str, default: float) -> float:
    """Parse float env var. Args: name/default. Returns: float. Raises: none."""
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    """Parse int env var. Args: name/default. Returns: int. Raises: none."""
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default


@dataclass(frozen=True)
class MarketDataPolicy:
    primary_source: str
    rest_fallback_enabled: bool
    rest_fallback_mode: str
    suppress_rest_before_ws: bool
    require_ws_spot_for_live: bool
    allow_synthetic_spot_in_live: bool
    nifty_internal_symbol: str
    nifty_zerodha_quote_symbol: str
    nifty_spot_token: int
    nifty_exchange: str
    startup_wait_for_ws_spot_seconds: float
    startup_spot_max_age_seconds: float
    block_live_if_no_ws_spot: bool
    quote_canonicalize_index_symbols: bool
    quote_mark_403_as_rest_degraded_only: bool
    quote_do_not_block_if_ws_healthy: bool
    log_throttle_quote_errors_seconds: float
    log_marketdata_decisions: bool

    @property
    def live_mode(self) -> bool:
        """Resolve LIVE mode. Args: none. Returns: bool. Raises: none."""
        return (
            str(os.getenv('EXECUTION_MODE', '')).strip().upper() == 'LIVE'
            or _env_bool('ENABLE_LIVE', False)
        )

    @classmethod
    def from_env(cls) -> 'MarketDataPolicy':
        """Build policy from env. Args: none. Returns: policy. Raises: none."""
        return cls(
            primary_source=os.getenv(
                'MARKETDATA_PRIMARY_SOURCE', 'websocket'
            ).strip().lower(),
            rest_fallback_enabled=_env_bool(
                'MARKETDATA_REST_FALLBACK_ENABLED', True
            ),
            rest_fallback_mode=os.getenv(
                'MARKETDATA_REST_FALLBACK_MODE', 'after_ws_degraded'
            ).strip().lower(),
            suppress_rest_before_ws=_env_bool(
                'MARKETDATA_SUPPRESS_REST_BEFORE_WS', True
            ),
            require_ws_spot_for_live=_env_bool(
                'MARKETDATA_REQUIRE_WS_SPOT_FOR_LIVE', True
            ),
            allow_synthetic_spot_in_live=_env_bool(
                'MARKETDATA_ALLOW_SYNTHETIC_SPOT_IN_LIVE', False
            ),
            nifty_internal_symbol=os.getenv(
                'NIFTY_INTERNAL_SYMBOL', 'NSE:NIFTY'
            ).strip(),
            nifty_zerodha_quote_symbol=os.getenv(
                'NIFTY_ZERODHA_QUOTE_SYMBOL', 'NSE:NIFTY 50'
            ).strip(),
            nifty_spot_token=_env_int('NIFTY_SPOT_TOKEN', 256265),
            nifty_exchange=os.getenv('NIFTY_EXCHANGE', 'NSE').strip(),
            startup_wait_for_ws_spot_seconds=_env_float(
                'STARTUP_WAIT_FOR_WS_SPOT_SECONDS', 15.0
            ),
            startup_spot_max_age_seconds=_env_float(
                'STARTUP_SPOT_MAX_AGE_SECONDS', 120.0
            ),
            block_live_if_no_ws_spot=_env_bool(
                'STARTUP_BLOCK_LIVE_IF_NO_WS_SPOT', True
            ),
            quote_canonicalize_index_symbols=_env_bool(
                'QUOTE_CANONICALIZE_INDEX_SYMBOLS', True
            ),
            quote_mark_403_as_rest_degraded_only=_env_bool(
                'QUOTE_MARK_403_AS_REST_DEGRADED_ONLY', True
            ),
            quote_do_not_block_if_ws_healthy=_env_bool(
                'QUOTE_DO_NOT_BLOCK_IF_WS_HEALTHY', True
            ),
            log_throttle_quote_errors_seconds=_env_float(
                'LOG_THROTTLE_QUOTE_ERRORS_SECONDS', 120.0
            ),
            log_marketdata_decisions=_env_bool('LOG_MARKETDATA_DECISIONS', True),
        )

    def is_internal_nifty(self, symbol: str) -> bool:
        """Check NIFTY aliases. Args: symbol. Returns: bool. Raises: none."""
        s = str(symbol or '').strip().upper()
        internal = self.nifty_internal_symbol.upper()
        return s in {
            'NIFTY',
            'NIFTY50',
            'NSE:NIFTY',
            'NSE:NIFTY50',
            'NSE:NIFTY 50',
            internal,
        }

    def to_broker_quote_symbol(self, symbol: str) -> str:
        """Map symbol for broker quote APIs. Args: symbol. Returns: str. Raises: none."""
        raw = str(symbol or '').strip()
        if not self.quote_canonicalize_index_symbols:
            return raw
        if self.is_internal_nifty(raw):
            return self.nifty_zerodha_quote_symbol
        s = raw.upper()
        if s in {'BANKNIFTY', 'NSE:BANKNIFTY', 'NSE:NIFTY BANK'}:
            return 'NSE:NIFTY BANK'
        if s in {'FINNIFTY', 'NSE:FINNIFTY', 'NSE:NIFTY FIN SERVICE'}:
            return 'NSE:NIFTY FIN SERVICE'
        return raw

    def quote_aliases(self, symbol: str) -> list[str]:
        """Return quote aliases. Args: symbol. Returns: list[str]. Raises: none."""
        raw = str(symbol or '').strip()
        if not raw:
            return []
        primary = self.to_broker_quote_symbol(raw)
        candidates = [
            primary,
            raw,
            primary.split(':', 1)[-1],
            raw.split(':', 1)[-1],
        ]
        out: list[str] = []
        for item in candidates:
            cleaned = str(item or '').strip()
            if cleaned and cleaned not in out:
                out.append(cleaned)
        return out

    def should_suppress_rest_quote(
        self,
        *,
        symbol: str,
        ws_started: bool,
        fallback_active: bool,
        explicit_allow_rest: bool | None = None,
    ) -> bool:
        """Decide REST quote suppression. Args: fields. Returns: bool. Raises: none."""
        if explicit_allow_rest is True:
            return False
        if explicit_allow_rest is False:
            return True
        if not self.live_mode:
            return False
        if not self.rest_fallback_enabled:
            return True
        if (
            self.is_internal_nifty(symbol)
            and self.suppress_rest_before_ws
            and not ws_started
            and not fallback_active
        ):
            return True
        if self.rest_fallback_mode == 'after_ws_degraded' and not fallback_active:
            return True
        return False

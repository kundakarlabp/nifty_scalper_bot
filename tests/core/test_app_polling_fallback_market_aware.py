"""Structural tests confirming the polling-fallback supervisor and the
DATA_WARMUP readiness gate are now market-session aware."""

from __future__ import annotations

import re
from pathlib import Path


_APP_PATH = Path(__file__).resolve().parents[2] / "src" / "nifty_scalper_bot" / "core" / "app.py"


def _source() -> str:
    return _APP_PATH.read_text(encoding="utf-8")


def test_polling_fallback_skipped_when_market_closed():
    """Args: none. Returns: None. Raises: AssertionError."""
    s = _source()
    # The supervisor must check market_open and short-circuit with
    # POLLING_FALLBACK_SKIPPED reason=market_closed.
    assert "polling_fallback_skipped_market_closed" in s
    assert "POLLING_FALLBACK_SKIPPED reason=market_closed" in s
    assert "market_open = is_market_open_now()" in s


def test_within_threshold_polling_skip_log_is_throttled():
    """Args: none. Returns: None. Raises: AssertionError."""
    s = _source()
    # The within_spot_stale_threshold log must run through log_throttled now.
    assert (
        re.search(
            r"polling_fallback_skipped:\{spot_symbol\}:within_spot_stale_threshold",
            s,
        )
        is not None
    )


def test_polling_fallback_activate_kept_for_open_market():
    """Args: none. Returns: None. Raises: AssertionError."""
    s = _source()
    assert "POLLING_FALLBACK_ACTIVATE reason=spot_stale" in s


def test_data_warmup_combines_market_closed_and_quote_denied():
    """Args: none. Returns: None. Raises: AssertionError."""
    s = _source()
    assert "data_warmup_reasons.append(\"broker_quote_access_denied\")" in s
    assert "data_warmup_reasons.append(\"market_closed\")" in s
    assert ',".join(data_warmup_reasons)' in s.replace(' ', '') or \
        '",".join(data_warmup_reasons)' in s


def test_market_session_state_logged_on_startup():
    """Args: none. Returns: None. Raises: AssertionError."""
    s = _source()
    assert "MARKET_SESSION_STATE state=%s" in s

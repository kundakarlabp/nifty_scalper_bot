"""Token-based historical data hydration for the Nifty scalper bot.

Fetches recent OHLC bars for an instrument token before the strategy starts
so that indicator engines have sufficient warm-up data.  Uses the broker
``historical_data()`` endpoint with a rolling look-back window to avoid the
Zerodha same-day API gap (same-day candles may not be available immediately).

Usage::

    from nifty_scalper_bot.core.hydration import assert_valid_token, hydrate

    assert_valid_token(token)
    bars = hydrate(kite, token)          # raises RuntimeError on < 50 bars
    bars = hydrate(kite, token, min_bars=20, lookback_days=5)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

LOGGER = logging.getLogger("nifty_scalper_bot.core.hydration")

# Conservative default: 3-day window ensures same-day gaps don't produce 0 bars
_DEFAULT_LOOKBACK_DAYS = 3
_DEFAULT_MIN_BARS = 50
_HARD_MIN_BARS = 20


def assert_valid_token(token: Any) -> int:
    """Raise RuntimeError when *token* is None, non-integer, or <= 0.

    Args:
        token: Raw token value to validate.

    Returns:
        int: Validated token as an integer.

    Raises:
        RuntimeError: When token is invalid.
    """
    if token is None:
        raise RuntimeError(
            "[FATAL] assert_valid_token: token is None — "
            "call InstrumentManager.load() before requesting tokens."
        )
    try:
        t = int(token)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"[FATAL] assert_valid_token: token={token!r} is not a valid integer"
        ) from exc
    if t <= 0:
        raise RuntimeError(
            f"[FATAL] assert_valid_token: token={t} must be a positive integer"
        )
    return t


def hydrate(
    kite: Any,
    token: int,
    *,
    min_bars: int = _DEFAULT_MIN_BARS,
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
    interval: str = "minute",
) -> list[dict[str, Any]]:
    """Fetch recent OHLC bars for *token* and verify we have enough data.

    Zerodha's historical API does not serve same-day candles reliably.
    Using a multi-day look-back window (default 3 days) avoids the
    common ``hydration_zero_bars`` failure seen when requesting only today.

    Args:
        kite: Broker client with a ``historical_data(token, from_dt, to_dt, interval)``
              method.  ``ZerodhaKiteClient`` satisfies this.
        token: Instrument token (positive integer).
        min_bars: Minimum number of bars required.  Raises ``RuntimeError``
                  when fewer bars are returned.  Defaults to 50.
        lookback_days: Number of calendar days to look back.  Defaults to 3.
        interval: Kite interval string (``"minute"``, ``"5minute"``, etc.).

    Returns:
        list[dict]: List of OHLC bar dicts, each with at least
        ``date``, ``open``, ``high``, ``low``, ``close``, ``volume``.

    Raises:
        RuntimeError: When token is invalid or fewer than *min_bars* bars
                      are returned (stops execution — do not swallow this).
    """
    token = assert_valid_token(token)
    effective_min = max(_HARD_MIN_BARS, int(min_bars))

    to_dt = datetime.now()
    from_dt = to_dt - timedelta(days=lookback_days)

    LOGGER.info(
        "Hydrating token=%d interval=%s from=%s to=%s min_bars=%d",
        token,
        interval,
        from_dt.strftime("%Y-%m-%d %H:%M"),
        to_dt.strftime("%Y-%m-%d %H:%M"),
        effective_min,
        extra={
            "event": "hydration_start",
            "token": token,
            "interval": interval,
            "lookback_days": lookback_days,
        },
    )

    try:
        data = kite.historical_data(token, from_dt, to_dt, interval)
    except Exception as exc:
        raise RuntimeError(
            f"[FATAL] Hydration API call failed for token={token}: {exc}"
        ) from exc

    if not data or len(data) < effective_min:
        bar_count = len(data) if data else 0
        raise RuntimeError(
            f"[FATAL] STOP: insufficient bars for token={token}. "
            f"Got {bar_count}, need {effective_min}. "
            "Extend lookback_days or check broker authentication."
        )

    LOGGER.info(
        "Hydration complete token=%d bars=%d",
        token,
        len(data),
        extra={
            "event": "hydration_complete",
            "token": token,
            "bars": len(data),
        },
    )
    return data


def hydrate_contracts(
    kite: Any,
    tokens: list[int],
    *,
    min_bars: int = _DEFAULT_MIN_BARS,
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
    interval: str = "minute",
    fail_fast: bool = False,
) -> dict[int, list[dict[str, Any]]]:
    """Hydrate multiple tokens, optionally raising on the first failure.

    Args:
        kite: Broker client with ``historical_data()`` method.
        tokens: List of positive instrument tokens.
        min_bars: Minimum bars required per token.
        lookback_days: Look-back window in calendar days.
        interval: Kite interval string.
        fail_fast: If True, raise on first failure. If False, log warnings
            and continue with remaining tokens.

    Returns:
        dict[int, list[dict]]: Mapping of token → OHLC bars.
            Only includes successfully hydrated tokens.

    Raises:
        RuntimeError: On any invalid token or insufficient bars if fail_fast=True.
    """
    results: dict[int, list[dict[str, Any]]] = {}
    failed_tokens: list[tuple[int, str]] = []
    
    for token in tokens:
        try:
            results[token] = hydrate(
                kite,
                token,
                min_bars=min_bars,
                lookback_days=lookback_days,
                interval=interval,
            )
        except RuntimeError as exc:
            error_msg = str(exc)
            failed_tokens.append((token, error_msg))
            if fail_fast:
                raise
            LOGGER.warning(
                "Hydration failed for token=%d: %s — skipping",
                token,
                error_msg,
                extra={"event": "hydration_token_failed", "token": token},
            )
    
    if failed_tokens:
        LOGGER.warning(
            "Hydration completed with %d/%d failures: failed_tokens=%s",
            len(failed_tokens),
            len(tokens),
            [t[0] for t in failed_tokens],
            extra={"event": "hydration_partial_complete", "failed_count": len(failed_tokens)},
        )
    
    LOGGER.info(
        "All tokens hydrated: count=%d",
        len(results),
        extra={"event": "hydration_all_complete", "count": len(results)},
    )
    return results

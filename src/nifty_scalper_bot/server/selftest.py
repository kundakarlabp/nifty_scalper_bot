"""Self-test HTTP endpoint exposing lightweight assessments."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Tuple

from fastapi import APIRouter, Request

from nifty_scalper_bot.data.assess_data import assess_datahub_fresh
from nifty_scalper_bot.diagnostics.assess import assess_suite

router = APIRouter(prefix="/selftest", tags=["selftest"])


@router.get("")
def selftest(request: Request) -> Dict[str, Any]:
    """
    Execute self-test probes against the running bot context.

    This endpoint must NEVER raise. All failures are reported as results.
    """

    ctx = _get_context(request)
    checks: List[Tuple[str, Callable[[], Tuple[bool, str, Dict[str, Any]]]]] = []

    if ctx is None:
        checks.append(("context_available", lambda: (False, "no_context", {})))
    else:
        hub = getattr(ctx, "data_hub", None)
        if hub is None:
            checks.append(("datahub_available", lambda: (False, "missing_datahub", {})))
        else:
            symbol, symbol_source = _resolve_symbol(ctx)
            poll_seconds = _resolve_poll_interval(ctx)
            adaptive_ms = max(2000, min(5000, int(poll_seconds * 1000 * 2.5)))

            def _datahub_fresh_check(
                hub: Any = hub,
                symbol: str = symbol,
                adaptive_ms: int = adaptive_ms,
                poll_seconds: float = poll_seconds,
                symbol_source: str = symbol_source,
            ) -> Tuple[bool, str, Dict[str, Any]]:
                try:
                    ok, detail, meta = assess_datahub_fresh(
                        hub,
                        symbol,
                        adaptive_ms,
                    )
                    meta = dict(meta or {})
                    meta.update(
                        {
                            "poll_seconds": poll_seconds,
                            "adaptive_ms": adaptive_ms,
                            "symbol_source": symbol_source,
                        }
                    )
                    return ok, detail, meta
                except Exception as exc:
                    # NEVER let self-test crash the server
                    return (
                        False,
                        "exception",
                        {
                            "error": type(exc).__name__,
                            "message": str(exc),
                            "symbol": symbol,
                            "poll_seconds": poll_seconds,
                            "adaptive_ms": adaptive_ms,
                        },
                    )

            checks.append(("datahub_fresh", _datahub_fresh_check))

    results = assess_suite(checks)
    return {
        "ok": all(result.ok for result in results),
        "results": [result.__dict__ for result in results],
    }


def _get_context(request: Request) -> Any:
    ctx_getter = getattr(request.app.state, "ctx_getter", None)
    if callable(ctx_getter):
        try:
            return ctx_getter()
        except Exception:
            return None
    return getattr(request.app.state, "bot_context", None)


def _resolve_symbol(ctx: Any) -> Tuple[str, str]:
    symbol = getattr(ctx, "spot_symbol", None)
    if symbol:
        return str(symbol), "spot_symbol"

    config = getattr(ctx, "config", None)
    if config is not None:
        symbols = getattr(config, "symbols", None)
        if isinstance(symbols, Iterable) and not isinstance(symbols, (str, bytes)):
            for candidate in symbols:
                if candidate:
                    return str(candidate), "config.symbols"
        if isinstance(symbols, (str, bytes)) and symbols:
            return str(symbols), "config.symbols"

    return "NIFTY", "fallback"


def _resolve_poll_interval(ctx: Any) -> float:
    poll_attr = getattr(ctx, "poll_interval_seconds", None)
    if poll_attr:
        try:
            return float(poll_attr)
        except (TypeError, ValueError):
            pass

    streamer = getattr(ctx, "streamer", None)
    interval = getattr(streamer, "_interval_s", None)
    if interval:
        try:
            return float(interval)
        except (TypeError, ValueError):
            pass

    return 0.7


__all__ = ["router", "selftest"]

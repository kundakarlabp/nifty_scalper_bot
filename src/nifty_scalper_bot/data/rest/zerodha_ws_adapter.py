"""Adapter for constructing a :class:`kiteconnect.KiteTicker` safely."""

from __future__ import annotations

from collections.abc import Sequence
from functools import wraps
from typing import Callable

from nifty_scalper_bot.utils.errors import ConfigurationError

try:  # pragma: no cover - optional dependency guard
    from kiteconnect import KiteTicker
except Exception:  # pragma: no cover - optional dependency guard
    KiteTicker = None  # type: ignore[assignment]


_HANDSHAKE_HEADERS: dict[str, str] = {
    "Origin": "https://kite.zerodha.com",
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/123.0 Safari/537.36"
    ),
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}
_PATCH_SENTINEL = "_nsb_header_injected"


def _sanitize_access_token(token: str | None) -> str:
    """Return Zerodha websocket token without the optional ``user:`` prefix."""

    value = (token or "").strip()
    if ":" in value:
        value = value.split(":", 1)[-1].strip()
    return value


def build_kite_ticker(
    api_key: str,
    access_token: str,
    *,
    session_refresher: Callable[[], None] | None = None,
):
    """Return a :class:`KiteTicker` configured with sanitized credentials."""

    if KiteTicker is None:  # pragma: no cover - runtime guard
        raise ConfigurationError("kiteconnect is not installed")

    token = _sanitize_access_token(access_token)
    if not api_key or not token:
        raise ConfigurationError("WebSocket credentials missing")

    ticker = KiteTicker(api_key, token)

    def _connect(*, threaded: bool = True) -> None:
        _install_ws_header_injector()
        # Do not pass custom headers – Zerodha's SDK manages handshake details.
        try:
            ticker.connect(threaded=threaded)  # type: ignore[call-arg]
        except TypeError:
            ticker.connect()  # type: ignore[call-arg]

    setattr(ticker, "_nsb_connect", _connect)
    if session_refresher is not None:
        setattr(ticker, "refresh_session", session_refresher)
    return ticker


def _normalize_header(header: object | None) -> list[str]:
    if header is None:
        return []
    if isinstance(header, dict):
        return [f"{key}: {value}" for key, value in header.items()]
    if isinstance(header, str):
        return [header]
    if isinstance(header, Sequence):
        normalized: list[str] = []
        for item in header:
            if isinstance(item, str):
                normalized.append(item)
            elif isinstance(item, Sequence) and len(item) == 2:
                key, value = item
                normalized.append(f"{key}: {value}")
        return normalized
    return [str(header)]


def _merge_headers(existing: object | None) -> list[str]:
    header_list = _normalize_header(existing)
    header_map: dict[str, str] = {}
    for entry in header_list:
        if ":" not in entry:
            continue
        key, value = entry.split(":", 1)
        header_map[key.strip()] = value.strip()

    header_map.update(_HANDSHAKE_HEADERS)
    return [f"{key}: {value}" for key, value in header_map.items()]


def _install_ws_header_injector() -> None:
    try:  # pragma: no cover - optional dependency guard
        import websocket
    except Exception:  # pragma: no cover - optional dependency guard
        return

    ws_app = getattr(websocket, "WebSocketApp", None)
    if ws_app is None:  # pragma: no cover - runtime guard
        return

    if getattr(ws_app, _PATCH_SENTINEL, False):
        return

    original_init = ws_app.__init__

    @wraps(original_init)
    def _patched_init(self, url, header=None, *args, **kwargs):  # noqa: ANN002, ANN003
        merged_header = _merge_headers(header)
        return original_init(self, url, header=merged_header, *args, **kwargs)

    ws_app.__init__ = _patched_init  # type: ignore[assignment]
    setattr(ws_app, _PATCH_SENTINEL, True)


__all__ = ["build_kite_ticker", "_install_ws_header_injector"]

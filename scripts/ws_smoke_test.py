from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any

from nifty_scalper_bot.data.rest.zerodha_client import (
    ZerodhaKiteWebSocket,
    _sanitize_access_token,
)

TOKEN_TO_CHECK = 256265  # NIFTY 50 spot


def main() -> int:
    api_key = os.getenv("ZERODHA_API_KEY")
    access_token = os.getenv("ZERODHA_ACCESS_TOKEN")
    if not api_key or not access_token:
        print(
            "Missing ZERODHA_API_KEY or ZERODHA_ACCESS_TOKEN in environment.",
            file=sys.stderr,
        )
        return 1

    sanitized = _sanitize_access_token(access_token)
    print(f"API key prefix={api_key[:4]}*** token_len={len(sanitized)}")

    ws = ZerodhaKiteWebSocket(api_key=api_key, access_token=sanitized)
    ticks: list[dict[str, Any]] = []
    done = threading.Event()

    def _on_tick(tick: dict[str, Any]) -> None:
        if not isinstance(tick, dict):
            return
        ticks.append(dict(tick))
        token_val = tick.get("instrument_token")
        ltp = tick.get("last_price") or tick.get("ltp")
        print(f"Tick #{len(ticks)} token={token_val} ltp={ltp}")
        if len(ticks) >= 3:
            done.set()

    def _on_error(exc: Exception) -> None:
        print(f"WebSocket error: {exc}", file=sys.stderr)
        done.set()

    ws.on_tick = _on_tick
    ws.on_error = _on_error

    try:
        ws.connect(threaded=True)
    except Exception as exc:  # noqa: BLE001 - diagnostic output only
        print(f"Connect failed: {exc}", file=sys.stderr)
        return 2

    try:
        ws.subscribe([TOKEN_TO_CHECK])
        deadline = time.monotonic() + 30.0
        while not done.wait(0.5):
            if time.monotonic() >= deadline:
                print("Timed out waiting for ticks (30s).", file=sys.stderr)
                break
    finally:
        ws.disconnect()

    if len(ticks) >= 3:
        print("Received 3 ticks. Smoke test OK.")
        return 0

    print(f"Smoke test incomplete: received {len(ticks)} tick(s).", file=sys.stderr)
    return 3


if __name__ == "__main__":
    sys.exit(main())

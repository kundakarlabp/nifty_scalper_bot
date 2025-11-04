"""FastAPI application exposing Prometheus metrics."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Response

from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.metrics import Counter, ensure_multiproc_dir
from prometheus_client import (  # type: ignore[attr-defined]
    CONTENT_TYPE_LATEST,
    generate_latest,
)

LOGGER = get_logger(__name__)

ensure_multiproc_dir()

app = FastAPI()


ticks_received = Counter("ticks_received_total", "Ticks received from broker")
signals_total = Counter("signals_total", "Signals generated", ["side"])
orders_attempted = Counter(
    "orders_attempted_total",
    "Orders attempted",
    ["side", "type"],
)
orders_filled = Counter("orders_filled_total", "Orders filled", ["side"])
orders_blocked = Counter("orders_blocked_total", "Orders blocked", ["reason"])


@app.get("/metrics")
def metrics() -> Response:
    """Return the latest Prometheus metrics payload.

    Args:
        None.

    Returns:
        Response: HTTP response with metrics payload.

    Raises:
        HTTPException: If metric generation fails.
    """

    LOGGER.debug(
        "Entered metrics",
        extra={"event": "metrics_endpoint_enter"},
    )
    try:
        payload = generate_latest()
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "Failure in metrics: %s",
            exc,
            extra={"event": "metrics_endpoint_error"},
            exc_info=exc,
        )
        raise HTTPException(status_code=500, detail="metrics unavailable") from exc
    return Response(payload, media_type=CONTENT_TYPE_LATEST)

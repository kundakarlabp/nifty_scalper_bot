from __future__ import annotations

import logging

from src.server.logging_utils import SimpleLogGate

_DECISION_GATE: SimpleLogGate = SimpleLogGate(default_interval_seconds=5.0)


def emit_decision(payload: dict) -> None:
    summary = {
        "event": "decision",
        "action": payload.get("action"),
        "side": payload.get("side"),
        "strike": payload.get("strike"),
        "lots": payload.get("lots"),
        "rr": round(payload.get("rr", 0), 2) if payload.get("rr") else None,
        "score": round(payload.get("score", 0), 2) if payload.get("score") else None,
        "atr_pct": round(payload.get("atr_pct", 0), 3)
        if payload.get("atr_pct")
        else None,
        "reason": payload.get("reason_block") or payload.get("reason"),
    }
    extras: dict[str, object] = {"summary": summary}
    label = payload.get("label")
    if label is not None:
        extras["label"] = label
    reason_block = payload.get("reason_block") or payload.get("reason")
    if reason_block is not None:
        extras["reason_block"] = reason_block
    plan_summary = payload.get("plan_summary")
    if isinstance(plan_summary, dict):
        extras["plan"] = plan_summary
    logger = logging.getLogger("decision")
    try:
        # Throttle noisy 'decision' logs; keep one every ~5s
        if _DECISION_GATE.ok("decision"):
            logger.info("decision", extra=extras)
    except Exception:
        # Fallback: still log if gating fails
        logger.info("decision", extra=extras)


def emit_micro(depth_ok: bool | None, spread_pct: float | None) -> None:
    logging.getLogger("micro").info(
        {
            "event": "micro",
            "depth_ok": depth_ok,
            "spread_pct": (
                round(spread_pct * 100.0, 3) if isinstance(spread_pct, (int, float)) else None
            ),
        }
    )


def emit_debug(event: str, payload: dict) -> None:
    logging.getLogger("debug").debug({"event": event, **payload})

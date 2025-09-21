from __future__ import annotations

import logging
import os

from .log_filters_compact import (
    compact_extra,
    summarize_decision,
    summarize_micro,
)

LOG = logging.getLogger("structured")


def emit(event: str, **fields) -> None:
    LOG.info(event, extra={"extra": {"event": event, **fields}})


def emit_decision(payload: dict, level: str | None = None) -> None:
    summary = summarize_decision(payload)
    lvl_raw = level if level is not None else os.getenv("DECISION_LOG_LEVEL", "INFO")
    lvl = str(lvl_raw).upper()
    fn = getattr(LOG, lvl.lower(), LOG.info)
    record_extra: dict[str, object] = {"extra": summary}
    if summary.get("label") is not None:
        record_extra["label"] = summary["label"]
    reason_value = summary.get("reason_block")
    if reason_value is not None:
        record_extra["reason_block"] = reason_value
        record_extra.setdefault("reason", reason_value)
    plan_snapshot = payload.get("plan")
    if plan_snapshot is not None:
        record_extra["plan"] = plan_snapshot
    fn("decision", extra=record_extra)


def emit_micro(payload: dict) -> None:
    LOG.info("micro", extra={"extra": summarize_micro(payload)})


def emit_debug_full(event: str, payload: dict) -> None:
    # Full JSON dump for /why or DEBUG runs
    LOG.debug(event, extra={"extra": {"event": event, **compact_extra(payload)}})

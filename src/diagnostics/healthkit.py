# src/diagnostics/healthkit.py
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass
class HealthItem:
    name: str
    ok: bool
    hint: str | None = None
    detail: str | None = None
    severity: str = "info"  # info|warn|error


def to_dict(
    items: List[HealthItem],
    *,
    last_signal: dict | None = None,
    meta: dict | None = None,
) -> Dict[str, Any]:
    return {
        "ok": all(x.ok for x in items),
        "checks": [asdict(x) for x in items],
        "last_signal": bool(last_signal),
        "meta": meta or {},
    }


def render_compact(items: List[HealthItem]) -> str:
    bullets = []
    for x in items:
        dot = "🟢" if x.ok else "🔴"
        bullets.append(f"{dot} {x.name}")
    head = "✅ Flow looks good" if all(i.ok for i in items) else "❗ Flow has issues"
    return head + "\n" + " · ".join(bullets)


def render_detailed(items: List[HealthItem], *, last_signal_present: bool) -> str:
    lines = ["🔍 Full system check"]
    for x in items:
        dot = "🟢" if x.ok else "🔴"
        extra = x.hint or x.detail or ""
        lines.append(f"{dot} {x.name}" + (f" — {extra}" if extra else ""))
    lines.append(f"📈 last_signal: {'present' if last_signal_present else 'none'}")
    return "\n".join(lines)


def snapshot_pipeline() -> Dict[str, Any]:
    """Return a diagnostic snapshot summarizing pipeline health."""

    from src.diagnostics.metrics import metrics, runtime_metrics
    from src.strategies.runner import StrategyRunner

    snapshot: Dict[str, Any] = {
        "ts": time.time(),
        "loop": metrics.snapshot(),
        "runtime": runtime_metrics.snapshot(),
    }

    runner = StrategyRunner._SINGLETON
    if runner is None:
        return snapshot

    runner_info: Dict[str, Any] = {
        "ready": bool(getattr(runner, "ready", False)),
        "paused": bool(getattr(runner, "_paused", False)),
    }
    try:
        runner_info["status"] = runner.get_status_snapshot()
    except Exception as exc:  # pragma: no cover - defensive
        runner_info["status_error"] = str(exc)
    snapshot["runner"] = runner_info

    try:
        health = runner.health_check()
    except Exception as exc:  # pragma: no cover - defensive
        health = {"error": str(exc)}
    snapshot["health"] = health

    data_source = getattr(runner, "data_source", None)
    if data_source is not None:
        data_info: Dict[str, Any] = {
            "last_tick_ts": getattr(data_source, "last_tick_ts", None),
        }
        tokens_fn = getattr(data_source, "current_tokens", None)
        if callable(tokens_fn):
            try:
                data_info["current_tokens"] = tokens_fn()
            except Exception:  # pragma: no cover - defensive
                data_info["current_tokens"] = None
        snapshot["data_source"] = data_info

    executor = getattr(runner, "order_executor", None)
    if executor is not None:
        queues = getattr(executor, "_queues", None)
        queue_depth = None
        if isinstance(queues, dict):
            try:
                queue_depth = sum(len(q) for q in queues.values())
            except Exception:  # pragma: no cover - defensive
                queue_depth = None
        exec_info: Dict[str, Any] = {
            "queue_depth": queue_depth,
            "cb_orders": getattr(getattr(executor, "cb_orders", None), "state", None),
        }
        snapshot["executor"] = exec_info

    return snapshot

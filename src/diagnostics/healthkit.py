# src/diagnostics/healthkit.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

@dataclass
class HealthItem:
    name: str
    ok: bool
    hint: str | None = None
    detail: str | None = None
    severity: str = "info"  # info|warn|error

def to_dict(items: List[HealthItem], *, last_signal: dict | None = None, meta: dict | None = None) -> Dict[str, Any]:
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

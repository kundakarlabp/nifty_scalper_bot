"""Display helpers for runtime install-proof status."""

from __future__ import annotations

from typing import Any, Mapping

_REQUIRED_FLAGS = (
    ("market_data_manager_hardened", "MDM"),
    ("websocket_hardened", "WebSocket"),
    ("datahub_synthetic_guard_installed", "DataHub synthetic guard"),
    ("polling_failover_runtime_patch_installed", "Polling failover"),
    ("core_app_import_hook_installed", "core.app hook"),
    ("datahub_import_hook_installed", "DataHub hook"),
)


def install_proof_display(proof: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a compact UI-ready summary for install proof."""

    if not isinstance(proof, Mapping) or not proof:
        return {
            "label": "UNKNOWN",
            "css": "warn",
            "all_installed": None,
            "missing": [],
            "hook_counts": {},
        }
    missing = [label for key, label in _REQUIRED_FLAGS if not bool(proof.get(key))]
    hook_counts = dict(proof.get("import_hook_counts") or {}) if isinstance(proof.get("import_hook_counts"), Mapping) else {}
    all_installed = bool(proof.get("all_required_installed")) and not missing
    if all_installed:
        label = "ALL HARDENING INSTALLED"
        css = "ok"
    elif missing:
        label = "HARDENING INCOMPLETE"
        css = "bad"
    else:
        label = "CHECK HARDENING"
        css = "warn"
    return {
        "label": label,
        "css": css,
        "all_installed": all_installed,
        "missing": missing,
        "hook_counts": hook_counts,
    }


__all__ = ["install_proof_display"]

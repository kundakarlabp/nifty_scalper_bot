#!/usr/bin/env python3
"""Load and validate the canonical agent architecture manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_MANIFEST = Path("docs/architecture/agent_manifest.json")


def load_manifest(
    root: Path,
    path: Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    validation_root = root
    resolved = path if path.is_absolute() else root / path
    if not resolved.exists() and not path.is_absolute():
        validation_root = Path(__file__).resolve().parents[1]
        resolved = validation_root / path

    payload = json.loads(resolved.read_text(encoding="utf-8"))
    validate_manifest(validation_root, payload)
    return payload


def validate_manifest(root: Path, payload: dict[str, Any]) -> None:
    if payload.get("version") != 1:
        raise ValueError("agent architecture manifest version must be 1")

    owners = payload.get("owners")
    if not isinstance(owners, list) or not owners:
        raise ValueError("agent architecture manifest requires owners")

    seen_owner_ids: set[str] = set()
    for owner in owners:
        owner_id = str(owner.get("id") or "").strip()
        if not owner_id or owner_id in seen_owner_ids:
            raise ValueError(f"invalid or duplicate owner id: {owner_id!r}")
        seen_owner_ids.add(owner_id)
        owner_path = str(owner.get("owner") or "")
        if not owner_path or not (root / owner_path).exists():
            raise ValueError(f"missing owner path for {owner_id}: {owner_path}")

    areas = payload.get("validation_areas")
    if not isinstance(areas, list) or not areas:
        raise ValueError("agent architecture manifest requires validation_areas")
    area_names = [str(item.get("name") or "") for item in areas]
    if len(area_names) != len(set(area_names)) or any(not name for name in area_names):
        raise ValueError("validation area names must be unique and non-empty")

    rules = payload.get("architecture_rules")
    if not isinstance(rules, list):
        raise ValueError("architecture_rules must be a list")
    rule_ids = [str(item.get("id") or "") for item in rules]
    if len(rule_ids) != len(set(rule_ids)) or any(not rule_id for rule_id in rule_ids):
        raise ValueError("architecture rule ids must be unique and non-empty")


def high_risk_markers(payload: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(item) for item in payload["high_risk_markers"])


def validation_rules(
    payload: dict[str, Any],
) -> tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...]:
    return tuple(
        (
            str(item["name"]),
            tuple(str(marker) for marker in item["markers"]),
            tuple(str(test) for test in item["tests"]),
        )
        for item in payload["validation_areas"]
    )


def architecture_rules(payload: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(dict(item) for item in payload["architecture_rules"])

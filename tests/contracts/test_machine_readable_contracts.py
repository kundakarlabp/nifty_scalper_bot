from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs" / "contracts" / "contract_manifest.json"

_TYPE_MAP = {
    "object": dict,
    "array": list,
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "null": type(None),
}


def _load(path: str) -> Any:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _matches_type(value: object, expected: str) -> bool:
    python_type = _TYPE_MAP[expected]
    if expected in {"integer", "number"} and isinstance(value, bool):
        return False
    return isinstance(value, python_type)


def _validate(schema: dict[str, Any], value: Any, path: str = "$") -> None:
    expected = schema.get("type")
    if isinstance(expected, list):
        assert any(_matches_type(value, item) for item in expected), (
            f"{path}: {type(value).__name__} not in {expected}"
        )
    elif isinstance(expected, str):
        assert _matches_type(value, expected), (
            f"{path}: {type(value).__name__} != {expected}"
        )

    if "enum" in schema:
        assert value in schema["enum"], f"{path}: {value!r} not in enum"

    if isinstance(value, dict):
        for key in schema.get("required", []):
            assert key in value, f"{path}: missing required key {key}"
        properties = schema.get("properties", {})
        for key, item in value.items():
            child = properties.get(key)
            if child is not None:
                _validate(child, item, f"{path}.{key}")
            elif schema.get("additionalProperties") is False:
                raise AssertionError(f"{path}: unexpected key {key}")
    elif isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            _validate(schema["items"], item, f"{path}[{index}]")


def test_contract_manifest_points_to_existing_owners_schemas_and_samples() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    ids: set[str] = set()

    for contract in manifest["contracts"]:
        assert contract["id"] not in ids
        ids.add(contract["id"])
        for field in ("schema", "sample", "owner"):
            assert (ROOT / contract[field]).exists(), contract


def test_contract_samples_match_machine_readable_schemas() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    for contract in manifest["contracts"]:
        schema = _load(contract["schema"])
        sample = _load(contract["sample"])
        assert schema["$schema"].endswith("/draft/2020-12/schema")
        assert schema["type"] == "object"
        assert set(schema["required"]).issubset(schema["properties"])
        _validate(schema, sample)


def test_trade_plan_sample_preserves_buy_geometry() -> None:
    sample = _load("tests/fixtures/contracts/trade_plan.json")

    assert sample["side"] == "BUY"
    assert sample["stop_loss"] < sample["entry_price"] < sample["take_profit"]
    assert sample["quantity"] % sample["resolved_lot_size"] == 0

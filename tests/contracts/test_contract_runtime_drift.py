from __future__ import annotations

import importlib
import json
from dataclasses import fields, is_dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs" / "contracts" / "contract_manifest.json"


def _load_symbol(value: str):
    module_name, symbol_name = value.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def test_machine_readable_contracts_do_not_drift_from_bound_dataclasses() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    bound = [
        contract
        for contract in manifest["contracts"]
        if contract.get("python_symbol")
    ]
    assert bound

    for contract in bound:
        runtime_type = _load_symbol(contract["python_symbol"])
        assert is_dataclass(runtime_type), contract["id"]

        runtime_fields = {item.name for item in fields(runtime_type)}
        schema = json.loads(
            (ROOT / contract["schema"]).read_text(encoding="utf-8")
        )
        schema_fields = set(schema["properties"])
        required_fields = set(schema["required"])

        assert schema_fields.issubset(runtime_fields), (
            f"{contract['id']} schema fields missing from runtime: "
            f"{sorted(schema_fields - runtime_fields)}"
        )
        assert required_fields.issubset(runtime_fields), (
            f"{contract['id']} required fields missing from runtime: "
            f"{sorted(required_fields - runtime_fields)}"
        )

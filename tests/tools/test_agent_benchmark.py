from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    path = ROOT / "scripts" / "agent_benchmark.py"
    spec = importlib.util.spec_from_file_location("agent_benchmark_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["agent_benchmark_test"] = module
    spec.loader.exec_module(module)
    return module


def test_historical_regression_benchmark_manifest_is_valid() -> None:
    module = _load_module()
    payload = module.load_manifest(ROOT)

    assert module.validate_manifest(ROOT, payload) == []
    assert len(payload["cases"]) >= 8
    assert len({case["id"] for case in payload["cases"]}) == len(payload["cases"])


def test_historical_regression_benchmark_deduplicates_pytest_targets() -> None:
    module = _load_module()
    payload = module.load_manifest(ROOT)

    targets = module.selected_targets(payload)

    assert targets
    assert len(targets) == len(set(targets))
    assert all(target.startswith("tests/") for target in targets)


def test_historical_regression_benchmark_can_select_one_case() -> None:
    module = _load_module()
    payload = module.load_manifest(ROOT)

    targets = module.selected_targets(payload, {"replay-parity"})

    assert targets == ["tests/backtests/test_replay_parity.py"]

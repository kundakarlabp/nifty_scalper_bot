from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    path = ROOT / "scripts" / "agent_test_impact.py"
    spec = importlib.util.spec_from_file_location("agent_test_impact_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_module_name_maps_source_path_to_import_path() -> None:
    module = _load_module()

    assert (
        module.module_name("src/nifty_scalper_bot/data/market_data_manager.py")
        == "nifty_scalper_bot.data.market_data_manager"
    )
    assert (
        module.module_name("src/nifty_scalper_bot/execution/__init__.py")
        == "nifty_scalper_bot.execution"
    )
    assert module.module_name("tests/test_example.py") is None


def test_impacted_tests_selects_direct_import_and_colocated_test(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path
    direct = root / "tests" / "data" / "test_quote_contract.py"
    colocated = root / "tests" / "data" / "test_market_data_manager_reconnect.py"
    unrelated = root / "tests" / "risk" / "test_limits.py"
    direct.parent.mkdir(parents=True)
    unrelated.parent.mkdir(parents=True)

    direct.write_text(
        "from nifty_scalper_bot.data.market_data_manager import MarketDataManager\n",
        encoding="utf-8",
    )
    colocated.write_text("def test_reconnect(): pass\n", encoding="utf-8")
    unrelated.write_text("def test_limits(): pass\n", encoding="utf-8")

    selected = module.impacted_tests(
        root,
        ("src/nifty_scalper_bot/data/market_data_manager.py",),
    )

    assert "tests/data/test_quote_contract.py" in selected
    assert "tests/data/test_market_data_manager_reconnect.py" in selected
    assert "tests/risk/test_limits.py" not in selected


def test_impacted_tests_includes_changed_test_file(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "tests" / "test_changed.py"
    path.parent.mkdir()
    path.write_text("def test_changed(): pass\n", encoding="utf-8")

    selected = module.impacted_tests(tmp_path, ("tests/test_changed.py",))

    assert selected == ("tests/test_changed.py",)

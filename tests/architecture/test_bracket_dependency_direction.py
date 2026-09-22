from __future__ import annotations

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
EXECUTION = ROOT / "src" / "nifty_scalper_bot" / "execution"

IMPLEMENTATION_FILES = (
    "hardened_bracket_manager.py",
    "canonical_bracket_manager.py",
    "ledger_bracket_manager.py",
    "runtime_bracket_manager.py",
)


def _tree(name: str) -> ast.Module:
    return ast.parse((EXECUTION / name).read_text(encoding="utf-8"))


def test_bracket_layers_depend_on_core_not_public_facade() -> None:
    public_facade = "nifty_scalper_bot.execution.bracket_manager"

    for name in IMPLEMENTATION_FILES:
        tree = _tree(name)
        absolute_imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        absolute_imports.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert public_facade not in absolute_imports, name

        relative_core_import = any(
            isinstance(node, ast.ImportFrom)
            and node.level == 1
            and node.module is None
            and any(alias.name == "bracket_core" for alias in node.names)
            for node in ast.walk(tree)
        )
        assert (
            relative_core_import
        ), f"{name} must import bracket_core directly, not through the public facade"

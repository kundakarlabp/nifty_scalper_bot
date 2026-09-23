"""Implementation layers must depend on the bracket core, not its public facade."""

import ast
from pathlib import Path

from nifty_scalper_bot.execution import bracket_core, bracket_manager


def test_bracket_implementation_layers_do_not_import_public_facade() -> None:
    execution = Path(__file__).resolve().parents[2] / "src/nifty_scalper_bot/execution"
    for name in (
        "hardened_bracket_manager",
        "canonical_bracket_manager",
        "ledger_bracket_manager",
        "runtime_bracket_manager",
    ):
        source = (execution / f"{name}.py").read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom):
                assert not (
                    node.module == "nifty_scalper_bot.execution"
                    and any(alias.name == "bracket_manager" for alias in node.names)
                ), name


def test_public_bracket_identity_is_stable() -> None:
    assert bracket_manager.BracketManager is bracket_manager.BoundBracketManager
    assert bracket_core.BracketManager in bracket_manager.BracketManager.__mro__


def test_bracket_implementation_layers_use_core_terminology() -> None:
    execution = Path(__file__).resolve().parents[2] / "src/nifty_scalper_bot/execution"
    for name in (
        "hardened_bracket_manager",
        "canonical_bracket_manager",
        "ledger_bracket_manager",
        "runtime_bracket_manager",
    ):
        source = (execution / f"{name}.py").read_text(encoding="utf-8")
        assert "_legacy" not in source, name

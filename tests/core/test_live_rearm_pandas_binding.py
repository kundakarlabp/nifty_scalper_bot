from __future__ import annotations

import ast
from pathlib import Path


def _app_tree() -> ast.Module:
    return ast.parse(
        Path("src/nifty_scalper_bot/core/app.py").read_text(encoding="utf-8")
    )


def test_live_rearm_readiness_binds_pandas_alias() -> None:
    tree = _app_tree()
    aliases = {
        alias.asname
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == "pandas"
    }
    assert aliases == {"pd"}


def test_runtime_readiness_timestamp_fallback_uses_bound_pd_alias() -> None:
    tree = _app_tree()
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "_recompute_and_push_runtime_readiness"
    )
    attributes = {
        node.attr
        for node in ast.walk(function)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "pd"
    }
    assert {"to_datetime", "isna", "Timestamp"}.issubset(attributes)
